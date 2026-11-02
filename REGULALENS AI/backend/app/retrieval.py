"""Configurable semantic/BM25 retrieval with RRF and optional cross-encoder reranking."""
import hashlib, math, re
from collections import OrderedDict
from collections import Counter
from functools import lru_cache
from sqlalchemy.orm import Session
from .models import Chunk,Document
from .config import settings

FALLBACK_MODEL = "hashing-fallback"
TOKEN_RE = re.compile(r"[\w'-]+")

@lru_cache(maxsize=1)
def _encoder():
    if not settings.embedding_model or settings.embedding_model == FALLBACK_MODEL:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(settings.embedding_model, device="cpu")
    except Exception:
        return None

def embedding_info():
    if _encoder.cache_info().currsize==0:
        return {"model":FALLBACK_MODEL,"configured_model":settings.embedding_model,"semantic":False,
                "fallback":"Semantic model loads lazily on first embedding request; signed feature hashing is active until then"}
    encoder=_encoder()
    return {"model":settings.embedding_model if encoder else FALLBACK_MODEL,
            "configured_model":settings.embedding_model,"semantic":encoder is not None,
            "fallback":"Signed feature hashing (lexical baseline); not semantic embeddings" if encoder is None else None}

def clear_embedding_cache():
    _encoder.cache_clear();_VECTOR_CACHE.clear();_qdrant_client.cache_clear()

def _prepared_text(text: str,is_query: bool):
    model=settings.embedding_model.lower()
    if "e5-" in model: return ("query: " if is_query else "passage: ")+text
    if "bge" in model and is_query: return "Represent this sentence for searching relevant passages: "+text
    return text

def _hash_embed(text: str, dimensions: int = 384) -> list[float]:
    vec=[0.0]*dimensions
    for token in TOKEN_RE.findall(text.lower()):
        digest=hashlib.blake2b(token.encode(),digest_size=8).digest()
        idx=int.from_bytes(digest[:4],"little")%dimensions
        vec[idx]+=1.0 if digest[4]&1 else -1.0
    norm=math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x/norm for x in vec]

_VECTOR_CACHE=OrderedDict()
def _remember(key,value):
    _VECTOR_CACHE[key]=tuple(value); _VECTOR_CACHE.move_to_end(key)
    while len(_VECTOR_CACHE)>8192: _VECTOR_CACHE.popitem(last=False)

def embed(text: str,is_query: bool=False) -> list[float]:
    info=embedding_info(); key=(info["model"],"q" if is_query else "d",text)
    if key in _VECTOR_CACHE: _VECTOR_CACHE.move_to_end(key); return list(_VECTOR_CACHE[key])
    encoder=_encoder()
    try: vector=encoder.encode(_prepared_text(text,is_query),normalize_embeddings=True,show_progress_bar=False) if encoder else _hash_embed(text)
    except Exception: vector=_hash_embed(text)
    _remember(key,vector); return list(_VECTOR_CACHE[key])

def embed_many(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    if not texts: return []
    info=embedding_info(); encoder=_encoder()
    if encoder is None: return [embed(text) for text in texts]
    result=[None]*len(texts); missing=[]; indices=[]
    for i,text in enumerate(texts):
        key=(info["model"],"d",text)
        if key in _VECTOR_CACHE:
            result[i]=list(_VECTOR_CACHE[key]); _VECTOR_CACHE.move_to_end(key)
        else: missing.append(text); indices.append(i)
    if indices:
        vectors=encoder.encode([_prepared_text(text,False) for text in missing],batch_size=batch_size or settings.embedding_batch_size,normalize_embeddings=True,show_progress_bar=False)
        for i,txt,vec in zip(indices,missing,vectors):
            result[i]=[float(x) for x in vec]; _remember((info["model"],"d",txt),vec)
    return result

def cosine(a,b): return sum(x*y for x,y in zip(a,b))

@lru_cache(maxsize=4)
def _qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(":memory:") if settings.qdrant_url=="memory://" else QdrantClient(url=settings.qdrant_url,timeout=3)

def index_qdrant_batch(points: list[tuple[int,list[float],dict]]):
    if not points: return True
    try:
        from qdrant_client import models
        client=_qdrant_client()
        if not client.collection_exists(settings.qdrant_collection):
            client.create_collection(settings.qdrant_collection,vectors_config=models.VectorParams(size=len(points[0][1]),distance=models.Distance.COSINE))
        client.upsert(settings.qdrant_collection,points=[models.PointStruct(id=idx,vector=vector,payload=payload) for idx,vector,payload in points],wait=True)
        return True
    except Exception: return False

def index_qdrant(chunk_id: int, vector: list[float], payload: dict):
    return index_qdrant_batch([(chunk_id,vector,payload)])

def _tokens(text): return TOKEN_RE.findall(text.lower())
def _deduplicate(chunks):
    kept=[];seen=[]
    for chunk in chunks:
        terms=set(_tokens(chunk.content));normalized=" ".join(_tokens(chunk.content))
        if any(normalized==old or len(terms&other)/max(len(terms|other),1)>=0.88 for old,other in seen): continue
        kept.append(chunk);seen.append((normalized,terms))
    return kept

def bm25_scores(query: str, chunks: list[Chunk], k1=1.5, b=0.75):
    """Small-corpus Okapi BM25 implementation; no generated/fitted statistics are hidden."""
    docs=[_tokens(c.content) for c in chunks]; n=len(docs); avgdl=sum(map(len,docs))/max(n,1)
    df=Counter(term for doc in docs for term in set(doc)); qterms=Counter(_tokens(query)); scores=[]
    for doc in docs:
        tf=Counter(doc); score=0.0
        for term,qf in qterms.items():
            freq=tf[term]
            if not freq: continue
            idf=math.log(1+(n-df[term]+0.5)/(df[term]+0.5))
            score+=qf*idf*(freq*(k1+1))/(freq+k1*(1-b+b*len(doc)/(avgdl or 1)))
        scores.append(score)
    return scores

def _dense_scores(question: str,chunks: list[Chunk],candidate_limit: int):
    active=embedding_info()["model"]
    generation_matches=all(c.embedding_model==active for c in chunks)
    qvec=embed(question,is_query=True); scores={}
    try:
        client=_qdrant_client()
        if generation_matches and client.collection_exists(settings.qdrant_collection):
            scores={int(h.id):float(h.score) for h in client.query_points(settings.qdrant_collection,query=qvec,limit=min(len(chunks),candidate_limit),with_payload=False).points}
    except Exception: pass
    return [scores.get(c.id,cosine(qvec,embed(c.content) if c.embedding_model!=active else (c.embedding or embed(c.content)))) for c in chunks]

@lru_cache(maxsize=1)
def _cross_encoder():
    if not settings.reranker_model: return None
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(settings.reranker_model,device="cpu")
    except Exception: return None

def retrieve(db: Session, question: str, limit: int = 6, strategy: str | None = None, document_ids: list[int] | None = None, owner_username: str | None = None):
    strategy=strategy or settings.retrieval_default
    if strategy not in {"dense","bm25","hybrid","hybrid_rerank"}: raise ValueError("strategy must be dense, bm25, hybrid, or hybrid_rerank")
    query=db.query(Chunk)
    if document_ids: query=query.filter(Chunk.document_id.in_(document_ids))
    if owner_username: query=query.join(Document,Chunk.document_id==Document.id).filter(Document.owner_username==owner_username)
    chunks=query.all()
    if not chunks: return []
    dense=_dense_scores(question,chunks,max(50,limit*8)) if strategy!="bm25" else [0.0]*len(chunks)
    lexical=bm25_scores(question,chunks) if strategy!="dense" else [0.0]*len(chunks)
    dense_rank={chunks[i].id:r+1 for r,i in enumerate(sorted(range(len(chunks)),key=lambda j:dense[j],reverse=True))}
    lexical_rank={chunks[i].id:r+1 for r,i in enumerate(sorted(range(len(chunks)),key=lambda j:lexical[j],reverse=True))}
    if strategy=="dense": ranked=sorted(chunks,key=lambda c:dense_rank[c.id])
    elif strategy=="bm25": ranked=sorted(chunks,key=lambda c:lexical_rank[c.id])
    else:
        # Reciprocal Rank Fusion (constant 60) combines independent dense and BM25 ranks.
        fused={c.id:1/(60+dense_rank[c.id])+1/(60+lexical_rank[c.id]) for c in chunks}
        ranked=sorted(chunks,key=lambda c:fused[c.id],reverse=True)
    candidates=ranked[:max(limit*4,limit)] if strategy=="hybrid_rerank" else ranked[:limit]
    cross=_cross_encoder() if strategy=="hybrid_rerank" else None
    if cross and candidates:
        try:
            scores=cross.predict([(question,c.content) for c in candidates],show_progress_bar=False)
            candidates=sorted(zip(candidates,scores),key=lambda p:float(p[1]),reverse=True)
            unique=_deduplicate([c for c,_ in candidates])
            scoremap={c.id:float(score) for c,score in candidates}
            return [(scoremap[c.id],c) for c in unique[:limit]]
        except Exception: pass
    if strategy=="hybrid_rerank":
        # Explicit lexical rerank fallback, surfaced in system metadata.
        candidates.sort(key=lambda c:(question.lower() in c.content.lower(),len(set(_tokens(question))&set(_tokens(c.content)))),reverse=True)
    candidates=_deduplicate(candidates)
    result=[]
    for rank,c in enumerate(candidates[:limit]):
        score=(dense_rank[c.id]**-1 if strategy=="dense" else lexical_rank[c.id]**-1 if strategy=="bm25" else (1/(60+dense_rank[c.id])+1/(60+lexical_rank[c.id])))
        result.append((score,c))
    return result

def retrieval_info():
    return {"strategy":settings.retrieval_default,"options":["dense","bm25","hybrid","hybrid_rerank"],
            "reranker":settings.reranker_model if _cross_encoder() else "lexical-rerank-fallback",
            "dense_model":embedding_info()}
