"""Compare available Gemini/Ollama providers on identical labeled questions."""
import argparse,asyncio,csv,json,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.llm import generate_with_provider,provider_status
async def run(path):
    rows=[json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    status=await provider_status();out=[]
    for row in rows:
        for provider in ("gemini","local-slm"):
            if provider=="gemini" and not status["gemini"]["configured"]: continue
            if provider=="local-slm" and not status["local_slm"]["available"]: continue
            start=time.perf_counter();error=None;answer=""
            try: answer=await generate_with_provider(provider,row["question"],row.get("evidence",""))
            except Exception as exc: error=str(exc)
            out.append({"question":row["question"],"provider":provider,"latency_ms":(time.perf_counter()-start)*1000,"answer":answer,"error":error,
                        "expected_answer_token_f1":None if not row.get("expected_answer") else _token_f1(answer,row["expected_answer"])})
    folder=Path(__file__).resolve().parents[1]/"reports";folder.mkdir(exist_ok=True);stamp=time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())
    report={"examples":len(rows),"provider_status":status,"rows":out,"note":"Answer score is token-overlap proxy; no human quality metrics inferred."}
    (folder/f"llm_benchmark_{stamp}.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    with (folder/f"llm_benchmark_{stamp}.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]) if out else ["question","provider","latency_ms","answer","error","expected_answer_token_f1"]);w.writeheader();w.writerows(out)
    return report
def _token_f1(a,b):
    aa=set(a.lower().split());bb=set(b.lower().split());p=len(aa&bb)/max(len(aa),1);r=len(aa&bb)/max(len(bb),1)
    return 2*p*r/max(p+r,1e-12)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("dataset");args=ap.parse_args();print(json.dumps(asyncio.run(run(args.dataset)),indent=2))
