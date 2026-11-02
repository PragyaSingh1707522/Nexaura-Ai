from contextlib import asynccontextmanager
from pathlib import Path
import base64, hashlib, hmac, json, time
from contextvars import ContextVar
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from sqlalchemy.orm import Session
from .config import settings
from .db import Base, engine, get_db
from . import models
from .schemas import AskRequest, VendorRequest, ReviewRequest
from .ingest import extract_text, chunk_text, validate_upload
from .retrieval import embed_many, retrieve, index_qdrant_batch, retrieval_info
from .llm import generate, provider_status
from .risk import assess
from .classify import classify
from .pii import scan_pii

current_actor=ContextVar("regulalens_actor",default="local-user")

def make_token(username: str):
    payload=base64.urlsafe_b64encode(json.dumps({"sub":username,"role":settings.auth_role,"exp":int(time.time())+8*3600}).encode()).decode().rstrip("=")
    signature=hmac.new(settings.auth_secret.encode(),payload.encode(),hashlib.sha256).hexdigest()
    return payload+"."+signature

def valid_token(token: str):
    try:
        payload,signature=token.split(".",1)
        expected=hmac.new(settings.auth_secret.encode(),payload.encode(),hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature,expected): return None
        data=json.loads(base64.urlsafe_b64decode(payload+"="*(-len(payload)%4)))
        if data.get("sub")==settings.auth_username and data.get("exp",0)>time.time(): return data
        return None
    except Exception: return None

def verify_password(password: str):
    stored=settings.auth_password_hash
    if stored:
        try:
            scheme,salt_hex,hash_hex=stored.split("$");salt=bytes.fromhex(salt_hex);expected=bytes.fromhex(hash_hex)
            actual=hashlib.scrypt(password.encode(),salt=salt,n=2**14,r=8,p=1,dklen=len(expected))
            return scheme=="scrypt" and hmac.compare_digest(actual,expected)
        except Exception:return False
    # Retained only for existing local MVP deployments; use AUTH_PASSWORD_HASH in new setups.
    return bool(settings.auth_password) and hmac.compare_digest(password,settings.auth_password)

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        protected=request.url.path.startswith("/api/") and request.url.path not in ("/api/health","/api/auth/login")
        if protected and (settings.auth_password or settings.auth_password_hash):
            if not settings.auth_username or not settings.auth_secret:
                return JSONResponse({"detail":"Authentication is misconfigured; set AUTH_USERNAME and AUTH_SECRET"},status_code=503)
            auth=request.headers.get("authorization","")
            claims=valid_token(auth[7:]) if auth.startswith("Bearer ") else None
            if not claims:
                return JSONResponse({"detail":"Authentication required"},status_code=401)
            if request.method in ("POST","PUT","PATCH","DELETE") and claims.get("role") not in ("admin","analyst"):
                return JSONResponse({"detail":"This role cannot modify workspace data"},status_code=403)
            if request.method=="DELETE" and claims.get("role")!="admin":
                return JSONResponse({"detail":"Administrator role required"},status_code=403)
            request.state.user=claims["sub"]
        else: request.state.user="local-user"
        actor_token=current_actor.set(getattr(request.state,"user","local-user"))
        try: return await call_next(request)
        finally: current_actor.reset(actor_token)

@asynccontextmanager
async def lifespan(app):
    Base.metadata.create_all(bind=engine)
    # Upgrade databases created by the earlier MVP without discarding existing rows.
    from sqlalchemy import inspect, text
    if "embedding_model" not in {c["name"] for c in inspect(engine).get_columns("chunks")}:
        with engine.begin() as conn: conn.execute(text("ALTER TABLE chunks ADD COLUMN embedding_model VARCHAR(255) DEFAULT 'hashing-fallback'"))
    if "owner_username" not in {c["name"] for c in inspect(engine).get_columns("documents")}:
        with engine.begin() as conn: conn.execute(text("ALTER TABLE documents ADD COLUMN owner_username VARCHAR(100) DEFAULT 'local-user'"))
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    yield

app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.add_middleware(AuthMiddleware)
cors_origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
if "*" in cors_origins:
    raise ValueError("CORS_ORIGINS cannot contain '*' when credentialed CORS is enabled")
app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def audit(db, action, details):
    db.add(models.AuditEvent(actor=current_actor.get(),action=action, details=details)); db.commit()

@app.get("/api/health")
def health(): return {"status":"ok","mode":"local","auth_required":bool(settings.auth_password or settings.auth_password_hash),"llm":"gemini" if settings.gemini_api_key else ("local-slm" if settings.local_slm_url else "extractive-fallback"),"retrieval":retrieval_info()}

@app.get("/api/models/status")
async def models_status():
    return {"embeddings":retrieval_info()["dense_model"],"retrieval":retrieval_info(),"llm":await provider_status()}

@app.post("/api/auth/login")
def login(req: dict):
    if not (settings.auth_password or settings.auth_password_hash): raise HTTPException(404,"Authentication is not configured")
    if not settings.auth_username or not settings.auth_secret: raise HTTPException(503,"Set AUTH_USERNAME and AUTH_SECRET to enable authentication")
    if not hmac.compare_digest(str(req.get("username","")),settings.auth_username) or not verify_password(str(req.get("password",""))):
        raise HTTPException(401,"Invalid username or password")
    return {"access_token":make_token(settings.auth_username),"token_type":"bearer","expires_in":28800}

@app.get("/api/documents")
def documents(request: Request,db: Session = Depends(get_db)):
    return [{"id":d.id,"filename":d.filename,"file_type":d.file_type,"status":d.status,"created_at":d.created_at,"chunks":len(d.chunks)} for d in db.query(models.Document).filter(models.Document.owner_username==request.state.user).order_by(models.Document.created_at.desc()).all()]

@app.post("/api/documents/upload")
async def upload(request: Request,file: UploadFile = File(...), db: Session = Depends(get_db)):
    name = Path(file.filename or "upload").name
    data = await file.read()
    if len(data) > 20 * 1024 * 1024: raise HTTPException(413, "File exceeds 20 MB limit")
    try:
        validate_upload(name,data)
        text = extract_text(name, data)
    except Exception as exc: raise HTTPException(400, str(exc))
    if not text.strip(): raise HTTPException(400, "No readable text found; scanned PDFs need OCR")
    doc = models.Document(filename=name, file_type=Path(name).suffix.lower(), text=text, status="indexed",owner_username=request.state.user)
    db.add(doc); db.flush()
    chunks = chunk_text(text)
    vectors=embed_many([content for content,_ in chunks])
    model_name=retrieval_info()["dense_model"]["model"]
    indexed = []
    for (content,page),vector in zip(chunks,vectors):
        row=models.Chunk(document_id=doc.id, content=content, page=page, embedding=vector,embedding_model=model_name); db.add(row); db.flush()
        indexed.append((row.id, vector, content, page))
    db.commit(); db.refresh(doc)
    qdrant_ok=index_qdrant_batch([(chunk_id,vector,{"document_id":doc.id,"filename":name,"page":page,"text":content[:1000]}) for chunk_id,vector,content,page in indexed])
    audit(db, "document.uploaded", {"document_id":doc.id,"filename":name,"chunks":len(chunks)})
    return {"id":doc.id,"filename":name,"status":doc.status,"chunks":len(chunks),"characters":len(text),"embedding_model":model_name,"qdrant_indexed":qdrant_ok}

@app.post("/api/documents/reindex")
def reindex_documents(db: Session = Depends(get_db)):
    rows=db.query(models.Chunk).all()
    vectors=embed_many([row.content for row in rows])
    model_name=retrieval_info()["dense_model"]["model"]
    points=[]
    for row,vector in zip(rows,vectors):
        row.embedding=vector; row.embedding_model=model_name
        doc=db.get(models.Document,row.document_id)
        points.append((row.id,vector,{"document_id":row.document_id,"filename":doc.filename if doc else "","page":row.page,"text":row.content[:1000]}))
    db.commit()
    # Replace the old collection so dimensions and model generations cannot be mixed.
    try:
        from .retrieval import _qdrant_client
        _qdrant_client.cache_clear();client=_qdrant_client()
        if client.collection_exists(settings.qdrant_collection): client.delete_collection(settings.qdrant_collection)
    except Exception: pass
    qdrant_ok=index_qdrant_batch(points)
    audit(db,"documents.reindexed",{"chunks":len(rows),"embedding_model":model_name,"qdrant_indexed":qdrant_ok})
    return {"chunks":len(rows),"embedding_model":model_name,"qdrant_indexed":qdrant_ok}

@app.delete("/api/documents/{document_id}")
def delete_document(document_id: int,request: Request, db: Session = Depends(get_db)):
    doc = db.get(models.Document, document_id)
    if not doc or doc.owner_username!=request.state.user: raise HTTPException(404, "Document not found")
    db.delete(doc); db.commit(); audit(db,"document.deleted",{"document_id":document_id}); return {"deleted":True}

@app.get("/api/documents/{document_id}/clauses")
def document_clauses(document_id: int,request: Request, db: Session = Depends(get_db)):
    doc=db.get(models.Document,document_id)
    if not doc or doc.owner_username!=request.state.user: raise HTTPException(404,"Document not found")
    import re
    segments=re.split(r"\n(?=(?:\d+(?:\.\d+)*[.)]?\s+|[A-Z][A-Z\s]{4,}$))",doc.text)
    clauses=[{"text":part.strip(),**classify(part)} for part in segments if len(part.strip())>50]
    audit(db,"document.clauses_extracted",{"document_id":document_id,"clauses":len(clauses)})
    return {"document_id":document_id,"method":clauses[0]["method"] if clauses else "no-classifiable-clauses","clauses":clauses}

@app.get("/api/documents/{document_id}/pii-scan")
def document_pii_scan(document_id: int,request: Request, db: Session = Depends(get_db)):
    doc=db.get(models.Document,document_id)
    if not doc or doc.owner_username!=request.state.user: raise HTTPException(404,"Document not found")
    result=scan_pii(doc.text);audit(db,"document.pii_scanned",{"document_id":document_id,"matches":result["count"]})
    return result

@app.post("/api/classify")
def classify_clause(req: dict):
    text=str(req.get("text",""))
    if not text.strip(): raise HTTPException(422,"text is required")
    return classify(text)

@app.post("/api/assistant/ask")
async def ask(req: AskRequest,request: Request, db: Session = Depends(get_db)):
    strategy=req.model_dump().get("strategy") or settings.retrieval_default
    try: results = retrieve(db, req.question,strategy=strategy,document_ids=req.document_ids,owner_username=request.state.user)
    except ValueError as exc: raise HTTPException(422,str(exc))
    sources = [{"id":f"S{i+1}","document_id":c.document_id,"filename":db.get(models.Document,c.document_id).filename,"page":c.page,"snippet":c.content[:500],"score":round(float(score),4)} for i,(score,c) in enumerate(results)]
    # Evidence grounding: response citations are constrained to returned source IDs.
    evidence = "\n\n".join(f"[S{i+1}] {c.content[:1400]}" for i,(_,c) in enumerate(results))
    from .verification import verify_answer,evidence_sufficiency
    sufficiency=evidence_sufficiency(req.question,sources,semantic_vectors=retrieval_info()["dense_model"]["semantic"])
    if not sources or not sufficiency["sufficient"]:
        answer="I could not find supporting evidence in the indexed documents. Please upload relevant material or rephrase the question."
        provider="insufficient-evidence-guardrail"
    else:
        answer, provider = await generate(req.question, evidence)
    original_answer=answer
    allowed = {s["id"] for s in sources}
    import re
    cited = set(re.findall(r"\[(S\d+)\]", answer))
    invalid = cited - allowed
    if invalid: answer = re.sub(r"\[(S\d+)\]", lambda m: m.group(0) if m.group(1) in allowed else "", answer)
    review = None
    if sources and (not cited or invalid):
        item = models.Review(kind="answer", item_id=0); db.add(item); db.flush(); review = item.id
    verification=verify_answer(original_answer,sources)
    if verification["warnings"] and not review:
        item=models.Review(kind="answer",item_id=0);db.add(item);db.flush();review=item.id
    audit(db,"assistant.answered",{"question":req.question[:200],"provider":provider,"sources":len(sources),"review_id":review,"verification":verification["status"]})
    return {"answer":answer,"sources":sources,"provider":provider,"review_required":review is not None,"review_id":review,"verification":verification,"evidence_sufficiency":sufficiency,"retrieval":{"strategy":strategy,**retrieval_info()}}

@app.get("/api/vendors")
def vendors(db: Session = Depends(get_db)):
    return [{"id":v.id,"vendor_name":v.vendor_name,"industry":v.industry,"score":v.score,"level":v.level,"factors":v.factors,"created_at":v.created_at} for v in db.query(models.VendorAssessment).order_by(models.VendorAssessment.created_at.desc()).all()]

@app.post("/api/vendors/assess")
def assess_vendor(req: VendorRequest, db: Session = Depends(get_db)):
    result = assess(req.model_dump())
    row = models.VendorAssessment(vendor_name=req.vendor_name, industry=req.industry, score=result["score"], level=result["level"], factors=result["factors"])
    db.add(row); db.commit(); db.refresh(row); audit(db,"vendor.assessed",{"assessment_id":row.id,"level":row.level})
    return {"id":row.id,**result}

@app.get("/api/evaluations")
def evaluations(db: Session = Depends(get_db)):
    docs=db.query(models.Document).count(); chunks=db.query(models.Chunk).count(); events=db.query(models.AuditEvent).count()
    report_dir=Path(__file__).resolve().parents[1]/"reports";files=sorted(report_dir.glob("evaluation_*.json")) if report_dir.exists() else []
    latest=json.loads(files[-1].read_text(encoding="utf-8")) if files else None
    llm_files=sorted(report_dir.glob("llm_benchmark_*.json")) if report_dir.exists() else []
    latest_llm=json.loads(llm_files[-1].read_text(encoding="utf-8")) if llm_files else None
    model_dir=Path(__file__).resolve().parents[1]/"models"
    def read_report(name):
        path=model_dir/name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return {"mode":"live-system-counters","documents":docs,"indexed_chunks":chunks,"audit_events":events,"retrieval":retrieval_info(),
      "latest_retrieval_report":latest,"llm_benchmark_report":latest_llm,"vendor_model_report":read_report("vendor_risk_metrics.json"),"clause_model_report":read_report("clause_metrics.json"),
      "note":"Metrics are shown only when produced from the committed dataset or provided ground truth."}

@app.get("/api/reviews")
def reviews(db: Session = Depends(get_db)):
    return [{"id":r.id,"kind":r.kind,"item_id":r.item_id,"status":r.status,"reviewer":r.reviewer,"note":r.note,"created_at":r.created_at} for r in db.query(models.Review).order_by(models.Review.created_at.desc()).all()]

@app.patch("/api/reviews/{review_id}")
def review_update(review_id: int, req: ReviewRequest, db: Session = Depends(get_db)):
    row=db.get(models.Review,review_id)
    if not row: raise HTTPException(404,"Review not found")
    row.status=req.status; row.reviewer=req.reviewer; row.note=req.note; db.commit(); audit(db,"review.updated",{"review_id":review_id,"status":req.status}); return {"id":row.id,"status":row.status}

@app.get("/api/audit")
def audit_log(db: Session = Depends(get_db)):
    return [{"id":e.id,"action":e.action,"details":e.details,"created_at":e.created_at} for e in db.query(models.AuditEvent).order_by(models.AuditEvent.created_at.desc()).limit(100).all()]
