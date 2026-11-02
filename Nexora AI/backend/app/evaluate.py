"""Run retrieval and answer metrics on caller-supplied, human-labeled JSONL.

Fields per row: question, relevant_chunk_ids, optional expected_answer,
expected_evidence, document_ids. Never creates gold labels or reports empty data as scores.
"""
import asyncio,csv,json,math,time
from datetime import datetime,timezone
from pathlib import Path
from .db import SessionLocal
from .retrieval import retrieve
from .llm import generate
from .verification import verify_answer

STRATEGIES=["dense","bm25","hybrid","hybrid_rerank"]
def _f1(prediction,reference):
    p=set(prediction.lower().split());r=set(reference.lower().split())
    if not p or not r:return 0.0
    precision=len(p&r)/len(p);recall=len(p&r)/len(r)
    return 2*precision*recall/(precision+recall) if precision+recall else 0.0
def _retrieval_metrics(results,relevant,k):
    ids=[c.id for _,c in results];rel=set(relevant)
    if not rel:return {"recall":None,"mrr":None,"ndcg":None}
    top=ids[:k];recall=len(set(top)&rel)/len(rel)
    rr=next((1/(i+1) for i,cid in enumerate(ids) if cid in rel),0.0)
    dcg=sum(1/math.log2(i+2) for i,cid in enumerate(top) if cid in rel)
    ideal=sum(1/math.log2(i+2) for i in range(min(len(rel),k)))
    return {"recall":recall,"mrr":rr,"ndcg":dcg/ideal if ideal else 0.0}
async def _evaluate(path,k_values=(5,10),strategies=STRATEGIES,write=True):
    rows=[json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows: raise ValueError("Evaluation dataset has no rows")
    db=SessionLocal();per_strategy={};answers=[]
    try:
        for row in rows:
            for strategy in strategies:
                start=time.perf_counter();found=retrieve(db,row["question"],limit=max(k_values),strategy=strategy,document_ids=row.get("document_ids"));elapsed=(time.perf_counter()-start)*1000
                sources=[{"id":f"S{i+1}","filename":"","page":chunk.page,"snippet":chunk.content,"document_id":chunk.document_id} for i,(_,chunk) in enumerate(found)]
                evidence="\n\n".join(f"[S{i+1}] {chunk.content}" for i,(_,chunk) in enumerate(found))
                prompt_question=row["question"]
                if row.get("output_format")=="json":prompt_question += "\nReturn only a valid JSON object with answer and citations keys."
                gen_start=time.perf_counter();answer,provider=await generate(prompt_question,evidence);generation_ms=(time.perf_counter()-gen_start)*1000
                verification=verify_answer(answer,sources)
                relevant=row.get("relevant_chunk_ids",[])
                json_validity=None
                if row.get("output_format")=="json":
                    try: json.loads(answer);json_validity=True
                    except Exception: json_validity=False
                item={"question":row["question"],"strategy":strategy,"latency_ms":elapsed,"generation_latency_ms":generation_ms,"total_latency_ms":elapsed+generation_ms,"provider":provider,
                      "recall_at_k":{str(k):_retrieval_metrics(found,relevant,k)["recall"] for k in k_values},
                      "mrr":_retrieval_metrics(found,relevant,max(k_values))["mrr"],
                      "ndcg_at_k":{str(k):_retrieval_metrics(found,relevant,k)["ndcg"] for k in k_values},
                      "answer_token_f1":_f1(answer,row["expected_answer"]) if row.get("expected_answer") else None,
                      "groundedness_proxy":sum(c["supported_by_overlap_check"] for c in verification["claims"])/max(len(verification["claims"]),1) if verification["claims"] else (1.0 if not evidence else 0.0),
                      "citation_precision":verification["citation_precision"],
                      "expected_evidence_overlap":len(set(row.get("expected_evidence"," ").lower().split())&set(evidence.lower().split()))/max(len(set(row.get("expected_evidence"," ").lower().split())),1) if row.get("expected_evidence") else None,
                      "context_relevance":max((len(set(row["question"].lower().split())&set(s["snippet"].lower().split()))/max(len(set(row["question"].lower().split())),1) for s in sources),default=0.0),
                      "json_validity":json_validity,"answer":answer,"verification":verification}
                answers.append(item);per_strategy.setdefault(strategy,[]).append(item)
    finally: db.close()
    summary={}
    for strategy,items in per_strategy.items():
        summary[strategy]={"queries":len(items),"latency_ms_mean":sum(i["latency_ms"] for i in items)/len(items),"generation_latency_ms_mean":sum(i["generation_latency_ms"] for i in items)/len(items),"total_latency_ms_mean":sum(i["total_latency_ms"] for i in items)/len(items),
          "recall_at_k":{str(k):_mean([i["recall_at_k"][str(k)] for i in items]) for k in k_values},
          "mrr":_mean([i["mrr"] for i in items]),"ndcg_at_k":{str(k):_mean([i["ndcg_at_k"][str(k)] for i in items]) for k in k_values},
          "answer_token_f1":_mean([i["answer_token_f1"] for i in items]),"groundedness_proxy":_mean([i["groundedness_proxy"] for i in items]),
          "citation_precision":_mean([i["citation_precision"] for i in items]),"expected_evidence_overlap":_mean([i["expected_evidence_overlap"] for i in items]),
          "context_relevance":_mean([i["context_relevance"] for i in items]),"json_validity":_mean([i["json_validity"] for i in items])}
    report={"dataset":str(path),"examples":len(rows),"created_at":datetime.now(timezone.utc).isoformat(),"metrics_are_measured":True,
            "warning":"Answer-token F1 and lexical groundedness are proxies, not expert judgments or semantic entailment. JSON validity is measured only for rows that request JSON output.","summary":summary,"rows":answers}
    if write:
        out=Path(__file__).resolve().parents[1]/"reports";out.mkdir(exist_ok=True);stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (out/f"evaluation_{stamp}.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        with (out/f"evaluation_{stamp}.csv").open("w",newline="",encoding="utf-8") as f:
            fields=["question","strategy","latency_ms","generation_latency_ms","total_latency_ms","provider","recall_at_k","mrr","ndcg_at_k","answer_token_f1","groundedness_proxy","citation_precision","expected_evidence_overlap","context_relevance","json_validity"]
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for row in answers: writer.writerow({key:json.dumps(row[key]) if isinstance(row[key],(dict,list)) else row[key] for key in fields})
    return report
def _mean(values):
    vals=[x for x in values if x is not None]
    return sum(vals)/len(vals) if vals else None
def evaluate(path,k_values=(5,10),strategies=STRATEGIES,write=True):
    return asyncio.run(_evaluate(path,k_values,strategies,write))
