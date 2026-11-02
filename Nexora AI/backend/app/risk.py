"""Vendor model inference; trained artifacts disclose synthetic label provenance explicitly."""
from .config import settings
FEATURES = ["data_sensitivity","access_level","control_maturity","incident_history","criticality"]
WEIGHTS = {"data_sensitivity":0.24,"access_level":0.20,"control_maturity":0.24,"incident_history":0.16,"criticality":0.16}

def assess(values: dict) -> dict:
    try:
        import joblib
        from pathlib import Path
        artifact_dir=Path(__file__).resolve().parents[1]/"models"
        preferred=artifact_dir/("vendor_risk_xgboost.joblib" if settings.risk_model=="xgboost" else "vendor_risk_model.joblib")
        model_path=preferred if preferred.exists() else artifact_dir/"vendor_risk_model.joblib"
        if model_path.exists():
            bundle=joblib.load(model_path)
            model=bundle["model"] if isinstance(bundle,dict) else bundle
            features=bundle.get("features",FEATURES) if isinstance(bundle,dict) else FEATURES
            vector = [[values[k] for k in FEATURES]]
            prob = float(model.predict_proba(vector)[0][1])
            explanation_kind="global_feature_importance"
            try:
                import shap
                import numpy as np
                estimator=model.steps[-1][1] if hasattr(model,"steps") else model
                if hasattr(estimator,"feature_importances_"):
                    explainer=shap.TreeExplainer(estimator);shap_values=explainer.shap_values(vector)
                    if isinstance(shap_values,list): shap_values=shap_values[-1]
                    if getattr(shap_values,"ndim",2)==3: shap_values=shap_values[:,:,1]
                    contributions={k:round(float(v),4) for k,v in zip(features,shap_values[0])}
                else:
                    background=np.asarray(bundle.get("background",vector))
                    explainer=shap.Explainer(model.predict_proba,background);explanation=explainer(np.asarray(vector))
                    vals=explanation.values[0]
                    if getattr(vals,"ndim",1)>1: vals=vals[:,1]
                    contributions={k:round(float(v),4) for k,v in zip(features,vals)}
                explanation_kind="local_shap_contribution"
            except Exception:
                estimator=model.steps[-1][1] if hasattr(model,"steps") else model
                if hasattr(estimator,"feature_importances_"):
                    contributions={k:round(float(v),4) for k,v in zip(features,estimator.feature_importances_)}
                elif hasattr(estimator,"coef_"):
                    contributions={k:round(float(v),4) for k,v in zip(features,estimator.coef_[0])}
                else: contributions={k:None for k in features}
            score = round(prob*100,1)
            return {"score":score,"level":level(score),"model":bundle.get("model_name","trained-model") if isinstance(bundle,dict) else "trained-model",
                    "model_version":bundle.get("version","legacy") if isinstance(bundle,dict) else "legacy",
                    "label_source":bundle.get("label_source","unknown") if isinstance(bundle,dict) else "unknown",
                    "explanation_kind":explanation_kind,"factors":contributions,
                    "explanation":"Trained model probability. Synthetic-label models are not real-world vendor risk probabilities." if isinstance(bundle,dict) and bundle.get("test_is_synthetic") else "Trained model score; explanations are feature contributions, not causal effects."}
    except Exception:
        pass
    normalized = {k: (6-values[k] if k == "control_maturity" else values[k]) / 5 for k in FEATURES}
    score = round(sum(normalized[k]*WEIGHTS[k] for k in FEATURES)*100,1)
    factors = {k:{"weight":WEIGHTS[k],"contribution":round(normalized[k]*WEIGHTS[k]*100,1)} for k in FEATURES}
    return {"score":score,"level":level(score),"model":"transparent-rule-baseline","factors":factors,"explanation":"No labeled training dataset is configured. Score uses disclosed weighted controls and is not a learned probability."}

def level(score): return "high" if score >= 67 else ("medium" if score >= 34 else "low")
