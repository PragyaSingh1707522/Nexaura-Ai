"""Conservative PII pattern scan for ingestion review; not a complete DLP detector."""
import re
PATTERNS={"email":re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
          "us_ssn":re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
          "phone":re.compile(r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?!\w)")}
def scan_pii(text: str):
    matches=[]
    for kind,pattern in PATTERNS.items():
        for match in pattern.finditer(text): matches.append({"type":kind,"start":match.start(),"end":match.end()})
    return {"detector":"pattern-baseline","matches":matches,"count":len(matches),"warning":"Pattern scan can miss PII and can produce false positives."}
