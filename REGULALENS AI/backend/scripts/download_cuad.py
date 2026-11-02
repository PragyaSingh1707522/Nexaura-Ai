"""Download the official Atticus Project CUAD v1 JSON release."""
from pathlib import Path
from huggingface_hub import hf_hub_download
target=Path(__file__).resolve().parents[1]/"datasets";target.mkdir(parents=True,exist_ok=True)
downloaded=hf_hub_download(repo_id="theatticusproject/cuad",filename="CUAD_v1/CUAD_v1.json",repo_type="dataset")
out=target/"CUAD_v1.json";out.write_bytes(Path(downloaded).read_bytes())
print(f"Downloaded official CUAD v1 annotation JSON to {out}")
