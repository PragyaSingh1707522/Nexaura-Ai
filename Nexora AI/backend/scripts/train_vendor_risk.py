"""Generate explicitly synthetic vendor questionnaire data and train/evaluate three models.

Usage: python scripts/train_vendor_risk.py --seed 2026 --n 1500
The target describes the documented synthetic risk rubric, not observed vendor outcomes.
"""
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

import joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"datasets"; ART=ROOT/"models"
FEATURES=["data_sensitivity","access_level","control_maturity","incident_history","criticality"]
def generate_dataset(n,seed):
    rng=np.random.default_rng(seed)
    x=pd.DataFrame({"data_sensitivity":rng.integers(1,6,n),"access_level":rng.integers(1,6,n),
      "control_maturity":rng.integers(1,6,n),"incident_history":rng.integers(0,6,n),"criticality":rng.integers(1,6,n)})
    # Documented synthetic scoring rule: protection maturity reduces risk; interactions model
    # higher risk when sensitive data and privileged access combine. Noise avoids a perfect rule.
    raw=(0.20*x.data_sensitivity+0.17*x.access_level+0.19*(6-x.control_maturity)+
         0.18*x.incident_history/5+0.13*x.criticality+0.13*(x.data_sensitivity*x.access_level/25)*5)
    probability=1/(1+np.exp(-3.4*(raw/5.0-0.59)))
    probability=np.clip(probability*0.82+0.09,0.04,0.96)
    x["synthetic_high_risk"]=(rng.random(n)<probability).astype(int)
    x["label_source"]="synthetic_rule_plus_noise"
    return x
def metrics(model,x,y):
    y=np.asarray(y,dtype=int);p=model.predict_proba(x)[:,1]; pred=(p>=0.5).astype(int)
    return {"roc_auc":float(roc_auc_score(y,p)),"accuracy":float(accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),"confusion_matrix":confusion_matrix(y,pred).tolist()}
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--n",type=int,default=1500);parser.add_argument("--seed",type=int,default=2026);args=parser.parse_args()
    DATA.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    frame=generate_dataset(args.n,args.seed);frame.to_csv(DATA/"synthetic_vendor_risk.csv",index=False)
    trainval,test=train_test_split(frame,test_size=.2,random_state=args.seed,stratify=frame.synthetic_high_risk)
    train,val=train_test_split(trainval,test_size=.25,random_state=args.seed,stratify=trainval.synthetic_high_risk)
    train.to_csv(DATA/"vendor_train.csv",index=False);val.to_csv(DATA/"vendor_validation.csv",index=False);test.to_csv(DATA/"vendor_test.csv",index=False)
    models={"logistic_regression":make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500,class_weight="balanced")),
            "random_forest":RandomForestClassifier(n_estimators=300,min_samples_leaf=3,class_weight="balanced",random_state=args.seed),
            "xgboost":XGBClassifier(n_estimators=250,max_depth=3,learning_rate=.04,subsample=.85,colsample_bytree=.85,
                 reg_lambda=2,eval_metric="logloss",random_state=args.seed,n_jobs=4)}
    comparison={}
    for name,model in models.items():
        model.fit(train[FEATURES],train.synthetic_high_risk)
        comparison[name]={"validation":metrics(model,val[FEATURES],val.synthetic_high_risk)}
    winner=max(models,key=lambda name:comparison[name]["validation"]["roc_auc"])
    selected=models[winner]
    comparison[winner]["test_holdout"] = metrics(selected,test[FEATURES],test.synthetic_high_risk)
    version="synthetic-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    joblib.dump({"model":selected,"model_name":winner,"features":FEATURES,"version":version,
      "label_source":"synthetic_rule_plus_noise","test_is_synthetic":True,
      "background":train[FEATURES].sample(min(100,len(train)),random_state=args.seed).values.tolist()},ART/"vendor_risk_model.joblib")
    if "xgboost" in models:
        joblib.dump({"model":models["xgboost"],"model_name":"xgboost","features":FEATURES,"version":version+"-xgb",
          "label_source":"synthetic_rule_plus_noise","test_is_synthetic":True,
          "background":train[FEATURES].sample(min(100,len(train)),random_state=args.seed).values.tolist()},ART/"vendor_risk_xgboost.joblib")
    report={"model_version":version,"selected_model":winner,"n":args.n,"seed":args.seed,
      "split":{"train":len(train),"validation":len(val),"test":len(test)},"label_source":"synthetic_rule_plus_noise",
      "warning":"Metrics describe prediction of a synthetic rubric only; they are not real-world vendor risk performance.","comparison":comparison}
    (ART/"vendor_risk_metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
