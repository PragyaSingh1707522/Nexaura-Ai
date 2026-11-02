"""CUAD-trained contract model when present; transparent keyword fallback otherwise."""
from functools import lru_cache
from pathlib import Path
from .config import settings
@lru_cache(maxsize=1)
def _trained():
    try:
        import joblib
        base=Path(__file__).resolve().parents[1]/"models"
        path=base/("clause_classifier_semantic.joblib" if settings.clause_model=="sentence_transformer" else "clause_classifier.joblib")
        return joblib.load(path) if path.exists() else None
    except Exception: return None
@lru_cache(maxsize=1)
def _semantic_encoder(model_id):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id,device="cpu")
LABELS={"Data protection":["personal data","privacy","data subject","gdpr","retention"],"Security controls":["encryption","access control","vulnerability","incident response","security"],"Audit & assurance":["audit","independent assessment","evidence","certification"],"Business continuity":["continuity","disaster recovery","recovery time","availability"],"Subprocessors":["subprocessor","third party","downstream"]}
def classify(text: str):
    bundle=_trained()
    if bundle:
        model=bundle["model"]
        if bundle.get("type")=="sentence_transformer":
            vector=_semantic_encoder(bundle["embedding_model"]).encode([text],normalize_embeddings=True,show_progress_bar=False)
            probs=model.predict_proba(vector)[0]
        else: probs=model.predict_proba([text])[0]
        if getattr(probs,"ndim",1)>1: probs=probs[0]
        ranked=sorted(zip(bundle["labels"],probs),key=lambda x:float(x[1]),reverse=True)
        chosen=[{"label":label,"confidence":round(float(score),4)} for label,score in ranked[:5] if float(score)>=0.15]
        return {"label":chosen[0]["label"] if chosen else "Unclassified","confidence":chosen[0]["confidence"] if chosen else 0.0,
                "labels":chosen,"method":"trained-sentence-transformer-logistic-cuad" if bundle.get("type")=="sentence_transformer" else "trained-tfidf-logistic-cuad",
                "dataset_scope":"commercial contract review; regulatory transfer is unvalidated"}
    low=text.lower(); scores={name:sum(low.count(term) for term in terms) for name,terms in LABELS.items()}
    best=max(scores,key=scores.get)
    return {"label":best if scores[best] else "Unclassified","confidence":round(scores[best]/max(sum(scores.values()),1),2),"method":"keyword-baseline","scores":scores}
