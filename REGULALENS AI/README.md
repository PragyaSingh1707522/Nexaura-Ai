# RegulaLens AI

RegulaLens is a local-first regulatory intelligence and vendor risk workspace. It includes a FastAPI service, React/TypeScript dashboard, PostgreSQL and Qdrant Compose services, document ingestion, hybrid retrieval, grounded answering, review records, and transparent vendor risk scoring.

## Quick start

### Docker (PostgreSQL + Qdrant)

1. Copy `.env.example` to `.env` and set a strong `POSTGRES_PASSWORD`. Set `INSTALL_ML=true` to include CPU Sentence Transformers and SHAP (larger image); add `GEMINI_API_KEY` only if Gemini responses are desired.
2. Run `docker compose up --build`.
3. Open <http://localhost:5173>. API docs are at <http://localhost:8000/docs> and Qdrant diagnostics at <http://localhost:6333/dashboard>.

### Local development (no Docker required)

Backend (Python 3.11/3.12 recommended):

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Frontend in another terminal:

```powershell
cd frontend
npm install
npm run dev
```

The base install stays lightweight. For actual Sentence Transformer embeddings install `backend/requirements-ml.txt`; the configured BGE model downloads on first model use. If its package or model files cannot load, the API reports `hashing-fallback` and does not call that semantic retrieval. Qdrant is used when available; SQLite/PostgreSQL vectors remain the offline fallback. After changing `EMBEDDING_MODEL`, call `POST /api/documents/reindex` to rebuild vectors and Qdrant. Set `VITE_API_BASE_URL` to the backend origin (without `/api`); the SPA appends `/api` to its API requests. Docker Compose defaults browser traffic to `http://localhost:8000` and allows development origins on localhost/127.0.0.1 ports 5173 and 5174 through `CORS_ORIGINS`. Containers use Docker service names (`postgres`, `qdrant`) for backend service connections.

## Working features

- Upload PDF, DOCX, XLSX, TXT, Markdown, and CSV. Text PDFs work; scanned PDFs need OCR, which is not bundled. Upload limit is 20 MB.
- Split extracted text into overlapping passages and batch-embed. BGE Sentence Transformers are used when installed and loadable; a disclosed signed feature-hashing baseline remains available offline. Cached embeddings are keyed by model and query/document role.
- Choose dense, BM25, hybrid RRF, or hybrid with reranking using `strategy` on `/api/assistant/ask`; configure the default with `RETRIEVAL_DEFAULT`. `RERANKER_MODEL` enables a Sentence Transformers cross-encoder; otherwise the API discloses its lexical reranker fallback. Retrieval supports document ID filters, deduplicated source chunks, and page-level citations.
- Answers pass citation-ID and lexical evidence-overlap checks. These are review signals, not proof of entailment. Invalid source IDs are flagged and stripped; absent evidence results in a direct insufficient-evidence response.
- `/api/classify` and `/api/documents/{id}/clauses` use a CUAD-trained TF-IDF + one-vs-rest Logistic Regression classifier when the model artifact exists, otherwise the keyword baseline is explicitly identified. CUAD covers commercial contract review categories, not regulatory obligations; transfer performance is unvalidated.
- Vendor model training creates 1,500 synthetic questionnaire records with a documented noisy rubric, performs train/validation/test splits, compares Logistic Regression, Random Forest and XGBoost, and stores a versioned selected model. The report labels all metrics synthetic. Inference exposes SHAP values when supported; otherwise it reports model coefficients/importances.
- Evaluations compare dense, BM25, hybrid and reranked retrieval and produce JSON/CSV reports with Recall@5/10, MRR, nDCG, latency, answer-token F1, citation precision, context relevance, and a lexical groundedness proxy. Gold question data is caller supplied; no benchmark labels or scores are fabricated.
- Local Ollama Qwen availability, latency, and memory can be inspected and benchmarked against Gemini using the same question/evidence set. PII pattern scanning and prompt-injection-aware prompts are included.
- Audit events and human-review queue are stored in the database.

Optional single-workspace bearer authentication is enabled when `AUTH_USERNAME`, `AUTH_PASSWORD_HASH`, and a long random `AUTH_SECRET` are configured. Generate an scrypt password hash with `cd backend; python scripts/hash_password.py`. Roles are `admin`, `analyst`, or `viewer`; only admin can delete workspace data. Documents are associated with the authenticated workspace identity and document routes/retrieval apply that ownership filter. There is only one configured account, so this is not multi-user tenant isolation or a full identity provider.

## Retrieval evaluation

Provide a human-reviewed evaluation JSONL file (one row per line) with `question`, gold `relevant_chunk_ids`, and optionally `expected_answer`, `expected_evidence`, and `document_ids`:

```json
{"question":"How long is personal data retained?","relevant_chunk_ids":[12,15],"expected_answer":"The policy says 90 days.","expected_evidence":"retained for 90 days"}
```

Run `cd backend && python -c "from app.evaluate import evaluate; print(evaluate('../evaluation.jsonl'))"` after uploading the corpus. It writes timestamped JSON and CSV experiment reports. Token F1 and lexical groundedness are proxies, not expert judgments or semantic entailment. Do not compare reports across corpora without recording the corpus and model versions.

For identical-provider comparison, prepare JSONL rows with `question` and `evidence` (optionally `expected_answer`) and run `cd backend && python scripts/benchmark_llms.py ../llm_questions.jsonl`. It records actual provider availability, latency, answers, and token-overlap proxy.

No 100-question regulatory evaluation set is bundled: this workspace had no regulatory corpus or expert judgments, and authoring answer keys without those would fabricate evaluation labels. `backend/datasets/regulatory_eval.schema.json` defines the input contract; provide your reviewed questions and the runner writes measured JSON/CSV reports.

## Model training

Vendor risk training (reproducible synthetic labels only): `cd backend && python scripts/train_vendor_risk.py --n 1500 --seed 2026`. It writes `datasets/synthetic_vendor_risk.csv`, split CSVs, versioned XGBoost and selected-model bundles, and `models/vendor_risk_metrics.json`. API inference defaults to the validation-selected model; set `RISK_MODEL=xgboost` to explicitly use the separately saved XGBoost model. The target is generated from a disclosed rule plus noise; these results are not real vendor-risk accuracy. The validation winner is Logistic Regression, so the experiment does not support choosing XGBoost as the best synthetic model.

Clause model training uses official [CUAD v1](https://huggingface.co/datasets/theatticusproject/cuad), a manually annotated commercial-contract dataset. Download and train with `cd backend && pip install -r requirements-ml.txt && python scripts/download_cuad.py && python scripts/train_clause_classifier.py --cuad-json datasets/CUAD_v1.json --seed 2026 --include-semantic`. This trains the TF-IDF baseline and optional Sentence Transformer classifier and saves per-label confusion matrices plus precision/recall/macro-F1 reports. CUAD labels are not regulatory obligations, so these results do not establish regulatory classification quality. Dataset provenance and limitations are described by [The Atticus Project](https://www.atticusprojectai.org/cuad/).

## Configuration

See `.env.example`. Gemini is optional. The local provider expects an Ollama-compatible `/api/generate`, `/api/tags`, and `/api/ps` endpoint. No credentials are included. PyTorch/Transformers and models are optional; the app uses a labeled fallback if they are unavailable.

## Security and production notes

The app provides optional single-workspace login, but not multi-user roles or tenant isolation. Do not expose it to an untrusted network as-is. Add an identity provider, tenant isolation, upload malware scanning, secrets management, rate limiting, retention policies, and production-grade observability before deployment. Prompt injection text in retrieved passages is treated as untrusted in the model instruction, and invalid source IDs are removed; automated answer fact checking is not claimed. Human review is queued when a cited response is absent or citations fail validation.

## Development

```powershell
cd backend; pytest -q
cd ../frontend; npm test; npm run build
```

The CI workflow runs backend tests, frontend tests, and the frontend production build. See `backend/app/` for API, ingestion, retrieval, providers, risk scoring, and evaluation modules.
