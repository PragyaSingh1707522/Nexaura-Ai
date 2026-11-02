"""Citation identity and evidence-overlap checks; this is a guardrail, not formal entailment."""
import re
from collections import Counter

STOPWORDS={"the","a","an","and","or","of","to","in","on","for","is","are","was","were","be","by","with","as","at","it","this","that","from","must","shall","should","may","can","will","we","our","their","they","has","have","had","only"}
def terms(text): return {w.lower() for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]+",text) if w.lower() not in STOPWORDS}
def evidence_sufficiency(question: str,sources: list[dict],semantic_vectors: bool=False):
    query=terms(question)
    lexical=max((len(query&terms(s.get("snippet","")))/max(len(query),1) for s in sources),default=0.0)
    dense=0.0
    if semantic_vectors and sources:
        try:
            from .retrieval import embed,cosine
            q=embed(question,is_query=True);dense=max(cosine(q,embed(s.get("snippet",""))) for s in sources)
        except Exception: pass
    sufficient=lexical>=0.20 or (semantic_vectors and dense>=0.42)
    return {"sufficient":sufficient,"term_coverage":round(lexical,3),"dense_similarity":round(dense,3) if semantic_vectors else None,
            "method":"term coverage and optional semantic cosine thresholds","thresholds":{"term_coverage":0.20,"dense_similarity":0.42}}

def verify_answer(answer: str, sources: list[dict]) -> dict:
    valid={s["id"]:s for s in sources}; cited=re.findall(r"\[(S\d+)\]",answer)
    invalid=sorted(set(cited)-set(valid)); claims=[]
    invalid_pages=[];page_refs=[]
    for match in re.finditer(r"\b(?:p(?:age)?\.?\s*)(\d+)\b",answer,re.I):
        page=int(match.group(1));near=answer[max(0,match.start()-24):min(len(answer),match.end()+24)]
        ids=[x for x in re.findall(r"\[(S\d+)\]",near) if x in valid]
        correct=any(int(valid[x].get("page",-1))==page for x in ids)
        page_refs.append({"page":page,"citation_ids":ids,"valid":correct})
        if not correct: invalid_pages.append(page)
    for part in re.split(r"(?<=[.!?])\s+|\n+",answer):
        text=part.strip()
        if len(terms(text))<3 or re.fullmatch(r"\[S\d+\]",text): continue
        refs=[ref for ref in re.findall(r"\[(S\d+)\]",text) if ref in valid]
        claim_terms=terms(re.sub(r"\[S\d+\]","",text))
        support=max((len(claim_terms & terms(valid[ref].get("snippet","")))/max(len(claim_terms),1) for ref in refs),default=0.0)
        claims.append({"claim":text,"citations":refs,"support_overlap":round(support,3),"supported_by_overlap_check":support>=0.25})
    unsupported=[c for c in claims if not c["supported_by_overlap_check"]]
    warnings=[]
    if not sources: warnings.append("No retrieved source evidence; treat the response as insufficient evidence.")
    if not cited and sources: warnings.append("Answer does not contain a verifiable source citation.")
    if invalid: warnings.append("Answer contains source IDs that were not returned by retrieval.")
    if invalid_pages: warnings.append("One or more mentioned page numbers do not match a cited source page.")
    if unsupported: warnings.append("One or more claims have weak lexical support; human review is recommended.")
    citation_hits=sum(1 for x in cited if x in valid);citation_total=len(cited)
    return {"status":"review_required" if warnings else "checks_passed","citation_precision":round(citation_hits/max(citation_total,1),3) if cited else 0.0,
            "citations":cited,"invalid_citations":invalid,"page_references":page_refs,"invalid_page_references":invalid_pages,"claims":claims,"unsupported_claims":unsupported,"warnings":warnings,
            "method":"citation-id validation + lexical evidence overlap; not semantic entailment"}
