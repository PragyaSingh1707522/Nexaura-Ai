from app.ingest import chunk_text
from app.retrieval import embed, cosine
from app.risk import assess
from app.classify import classify
from app.config import settings
from app.retrieval import clear_embedding_cache

def test_chunking_preserves_page():
    chunks=chunk_text("[Page 3]\n" + "regulatory obligation "*100, size=120, overlap=20)
    assert chunks and all(page==3 for _,page in chunks)

def test_embedding_is_deterministic_and_normalized(monkeypatch):
    monkeypatch.setattr(settings,"embedding_model","hashing-fallback");clear_embedding_cache()
    a=embed("privacy encryption controls"); assert a==embed("privacy encryption controls")
    assert abs(cosine(a,a)-1)<1e-6

def test_risk_score_and_explanation():
    result=assess({"vendor_name":"Acme","industry":"SaaS","data_sensitivity":5,"access_level":5,"control_maturity":1,"incident_history":5,"criticality":5})
    assert 0<=result["score"]<=100 and result["model"]=="logistic_regression"
    assert result["label_source"]=="synthetic_rule_plus_noise" and result["explanation_kind"]=="local_shap_contribution"

def test_xgboost_artifact_can_be_selected(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings,"risk_model","xgboost")
    result=assess({"vendor_name":"Acme","industry":"SaaS","data_sensitivity":4,"access_level":4,"control_maturity":2,"incident_history":1,"criticality":4})
    assert result["model"]=="xgboost" and result["model_version"].endswith("-xgb")

def test_clause_classifier_is_transparent():
    result=classify("Personal data retention and GDPR requirements")
    assert result["method"].startswith("trained-")
    assert result["dataset_scope"].startswith("commercial contract")
