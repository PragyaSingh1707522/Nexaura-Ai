from .classify import classify

def extract_clauses(text: str):
    import re
    parts=re.split(r"\n(?=(?:\d+(?:\.\d+)*[.)]?\s+|[A-Z][A-Z\s]{4,}$))",text)
    return [{"text":p.strip(),**classify(p)} for p in parts if len(p.strip())>80]
