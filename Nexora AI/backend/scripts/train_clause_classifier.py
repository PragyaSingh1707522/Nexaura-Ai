"""Train contract clause snippet classifiers from the official CUAD v1 annotations.

Positive samples use annotated answer spans with surrounding text. Negative examples are
unannotated sentence windows sampled within the same contracts. Splits are by contract,
preventing text leakage across train/validation/test.
"""
import argparse,csv,json,re
from pathlib import Path
import sys
import joblib,numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,f1_score,multilabel_confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def parse_cuad(path,negative_windows_per_contract=16):
    data=json.loads(Path(path).read_text(encoding="utf-8")); examples=[]
    for contract in data.get("data",[]):
        title=contract.get("title","unknown"); positives=[]
        for paragraph in contract.get("paragraphs",[]):
            context=paragraph.get("context","")
            for qa in paragraph.get("qas",[]):
                question=qa.get("question","");match=re.search(r"related to\s*['\"]?([^'\"?.]+)",question,re.I)
                label=(match.group(1) if match else question).strip().rstrip(".?! ")
                for ans in qa.get("answers",[]):
                    answer=ans.get("text","");start=int(ans.get("answer_start",-1))
                    if not answer or start<0: continue
                    lo=max(0,start-350);hi=min(len(context),start+len(answer)+350)
                    positives.append((lo,hi));snippet=context[lo:hi].strip()
                    if len(snippet)>=40: examples.append({"text":snippet,"label":label,"group":title})
            # Unannotated windows serve only as negatives; overlapping any answer span are excluded.
            candidates=[]
            for sent in re.finditer(r"[^.!?\n]+(?:[.!?]|$)",context):
                lo,hi=sent.span();text=sent.group().strip()
                if len(text)>70 and not any(lo<pos_hi and hi>pos_lo for pos_lo,pos_hi in positives): candidates.append(text)
            if candidates:
                indexes=np.linspace(0,len(candidates)-1,min(negative_windows_per_contract,len(candidates)),dtype=int)
                for ix in indexes: examples.append({"text":candidates[int(ix)],"label":"NO_TARGET_CLAUSE","group":title})
    return examples
def metric(y,p,labels):
    pred=p>=.5
    return {"micro_precision":float(precision_score(y,pred,average="micro",zero_division=0)),"micro_recall":float(recall_score(y,pred,average="micro",zero_division=0)),"micro_f1":float(f1_score(y,pred,average="micro",zero_division=0)),"macro_precision":float(precision_score(y,pred,average="macro",zero_division=0)),"macro_recall":float(recall_score(y,pred,average="macro",zero_division=0)),"macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),"per_label_confusion_matrices":{name:mat.tolist() for name,mat in zip(labels,multilabel_confusion_matrix(y,pred))}}
def cap_by_label(indices,y,cap,seed):
    rng=np.random.default_rng(seed);picked=[]
    for col in range(y.shape[1]):
        positives=np.asarray(indices)[y[indices,col]==1]
        if len(positives): picked.extend(rng.choice(positives,size=min(cap,len(positives)),replace=False).tolist())
    return np.asarray(sorted(set(picked)),dtype=int)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--cuad-json",required=True);ap.add_argument("--seed",type=int,default=2026);ap.add_argument("--include-semantic",action="store_true");args=ap.parse_args()
    examples=parse_cuad(args.cuad_json);texts=[e["text"] for e in examples];groups=np.asarray([e["group"] for e in examples]);labels=[(e["label"],) for e in examples]
    mlb=MultiLabelBinarizer();y=mlb.fit_transform(labels)
    splitter=GroupShuffleSplit(n_splits=1,test_size=.3,random_state=args.seed);tr,rest=next(splitter.split(texts,y,groups))
    split2=GroupShuffleSplit(n_splits=1,test_size=.5,random_state=args.seed);va_rel,te_rel=next(split2.split([texts[i] for i in rest],y[rest],groups[rest]));va,te=rest[va_rel],rest[te_rel]
    model=Pipeline([("tfidf",TfidfVectorizer(max_features=60000,ngram_range=(1,2),min_df=2,sublinear_tf=True)),("classifier",OneVsRestClassifier(LogisticRegression(max_iter=500,class_weight="balanced",solver="liblinear")))])
    model.fit([texts[i] for i in tr],y[tr]);comparisons={"tfidf_logistic":{"validation":metric(y[va],model.predict_proba([texts[i] for i in va]),mlb.classes_),"test_holdout":metric(y[te],model.predict_proba([texts[i] for i in te]),mlb.classes_)}}
    out=ROOT/"models";out.mkdir(exist_ok=True);best_bundle={"type":"tfidf","model":model}
    if args.include_semantic:
        from sentence_transformers import SentenceTransformer
        from app.config import settings
        encoder=SentenceTransformer(settings.embedding_model,device="cpu");encoder.max_seq_length=128
        # The semantic model run is a bounded, label-stratified research experiment; the
        # TF-IDF baseline still uses the full train/validation/test partitions.
        sem_tr=cap_by_label(tr,y,24,args.seed);sem_va=cap_by_label(va,y,12,args.seed+1);sem_te=cap_by_label(te,y,12,args.seed+2)
        sem_ids=np.concatenate([sem_tr,sem_va,sem_te]);vectors=encoder.encode([texts[i] for i in sem_ids],batch_size=64,normalize_embeddings=True,show_progress_bar=True)
        vmap={idx:vectors[pos] for pos,idx in enumerate(sem_ids)}
        Xtr=np.asarray([vmap[i] for i in sem_tr]);Xva=np.asarray([vmap[i] for i in sem_va]);Xte=np.asarray([vmap[i] for i in sem_te])
        semantic_model=OneVsRestClassifier(LogisticRegression(max_iter=500,class_weight="balanced",solver="liblinear"))
        semantic_model.fit(Xtr,y[sem_tr]);comparisons["sentence_transformer_logistic"]={"validation":metric(y[sem_va],semantic_model.predict_proba(Xva),mlb.classes_),"test_holdout":metric(y[sem_te],semantic_model.predict_proba(Xte),mlb.classes_),"sample_counts":{"train":len(sem_tr),"validation":len(sem_va),"test":len(sem_te)}}
        semantic_bundle={"type":"sentence_transformer","model":semantic_model,"embedding_model":settings.embedding_model}
        joblib.dump({**semantic_bundle,"labels":list(mlb.classes_),"dataset":"CUAD v1"},out/"clause_classifier_semantic.joblib")
        if comparisons["sentence_transformer_logistic"]["validation"]["macro_f1"]>comparisons["tfidf_logistic"]["validation"]["macro_f1"]:best_bundle=semantic_bundle
    # Export metadata, not the full duplicated snippets, to keep the training output compact.
    best=max(comparisons,key=lambda name:comparisons[name]["validation"]["macro_f1"])
    if best=="sentence_transformer_logistic":best_bundle={"type":"sentence_transformer","model":semantic_model,"embedding_model":settings.embedding_model}
    joblib.dump({**best_bundle,"labels":list(mlb.classes_),"dataset":"CUAD v1","label_scope":"annotated contract clause snippets; not regulatory obligations"},out/"clause_classifier.joblib")
    report={"dataset":"CUAD v1","examples":len(texts),"unique_contracts":len(set(groups)),"labels":list(mlb.classes_),"seed":args.seed,"split":{"train":len(tr),"validation":len(va),"test":len(te),"grouped_by_contract":True},"selected_model":best,"comparisons":comparisons,"limitation":"CUAD annotations cover commercial contract review clauses. Transfer to regulatory language is not validated. Sentence-Transformer run uses label-stratified capped samples, while TF-IDF uses the complete partitions."}
    (out/"clause_metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
