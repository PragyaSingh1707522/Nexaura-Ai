from pathlib import Path
from io import BytesIO
from zipfile import BadZipFile,ZipFile

SUPPORTED={".pdf",".docx",".xlsx",".txt",".md",".csv"}
def validate_upload(filename: str,data: bytes):
    suffix=Path(filename).suffix.lower()
    if suffix not in SUPPORTED: raise ValueError("Supported formats: PDF, DOCX, XLSX, TXT and MD/CSV")
    if suffix==".pdf" and not data.lstrip().startswith(b"%PDF-"): raise ValueError("File content does not match PDF format")
    if suffix in (".docx",".xlsx"):
        if not data.startswith(b"PK"): raise ValueError("Office document is not a valid ZIP container")
        try:
            with ZipFile(BytesIO(data)) as archive:
                items=archive.infolist()
                if len(items)>5000 or sum(i.file_size for i in items)>200*1024*1024: raise ValueError("Office file archive exceeds safe extraction limits")
                if any(i.flag_bits&1 for i in items): raise ValueError("Encrypted Office archives are not supported")
                names={i.filename for i in items}
                required="word/document.xml" if suffix==".docx" else "xl/workbook.xml"
                if required not in names: raise ValueError("File content does not match Office document type")
        except BadZipFile as exc: raise ValueError("Office document ZIP is corrupt") from exc
    return True

def extract_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    validate_upload(filename,data)
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(data))
        return "\n".join(f"[Page {i+1}]\n{page.extract_text() or ''}" for i, page in enumerate(reader.pages)).strip()
    if suffix == ".docx":
        from docx import Document
        doc = Document(BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if suffix == ".xlsx":
        from openpyxl import load_workbook
        workbook = load_workbook(BytesIO(data), read_only=True, data_only=True)
        return "\n".join(f"[{sheet.title}]\n" + "\n".join(" | ".join(str(v or "") for v in row) for row in sheet.iter_rows(values_only=True)) for sheet in workbook.worksheets)
    if suffix in (".txt", ".md", ".csv"):
        return data.decode("utf-8-sig", errors="replace")
    raise ValueError("Supported formats: PDF, DOCX, XLSX, TXT and MD")

def chunk_text(text: str, size: int = 1100, overlap: int = 160) -> list[tuple[str, int]]:
    import re
    # Preserve explicit PDF page markers as citation metadata.
    pieces = re.split(r"(?=\[Page \d+\])", text)
    result = []
    for piece in pieces:
        match = re.match(r"\[Page (\d+)\]\s*", piece)
        page = int(match.group(1)) if match else 1
        body = piece[match.end():] if match else piece
        start = 0
        while start < len(body):
            end = min(start + size, len(body))
            content = body[start:end].strip()
            if content: result.append((content, page))
            if end == len(body): break
            start = end - overlap
    return result
