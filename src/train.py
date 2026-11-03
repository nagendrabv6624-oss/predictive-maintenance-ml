from __future__ import annotations
import argparse,json
from pathlib import Path
import joblib,matplotlib.pyplot as plt,numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,average_precision_score,classification_report,ConfusionMatrixDisplay,confusion_matrix,f1_score,precision_score,recall_score,roc_auc_score,RocCurveDisplay,PrecisionRecallDisplay
from sklearn.model_selection import StratifiedKFold,cross_validate,train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
ROOT=Path(__file__).resolve().parents[1]
NUM=["Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"]
CAT=["Type"]; FEATURES=NUM+CAT; TARGET="Machine failure"
def fetch_ai4i():
 from ucimlrepo import fetch_ucirepo
 ds=fetch_ucirepo(id=601)
 if TARGET not in ds.data.targets.columns: raise ValueError("UCI response missing Machine failure target")
 return ds.data.features.join(ds.data.targets[[TARGET]])
def load_data(path):
 f=pd.read_csv(path) if path else fetch_ai4i()
 missing=set(FEATURES+[TARGET])-set(f.columns)
 if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
 f=f[FEATURES+[TARGET]].copy()
 for c in NUM+[TARGET]: f[c]=pd.to_numeric(f[c],errors="coerce")
 f=f.dropna()
 if len(f)<100 or set(f[TARGET].unique())!={0,1}: raise ValueError("Need 100+ complete rows and both target classes")
 f[TARGET]=f[TARGET].astype(int)
 return f
def pipe(clf):
 numeric=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())])
 categorical=Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))])
 prep=ColumnTransformer([("numeric",numeric,NUM),("categorical",categorical,CAT)])
 return Pipeline([("preprocess",prep),("classifier",clf)])
def main():
 p=argparse.ArgumentParser();p.add_argument("--data",default=None);a=p.parse_args()
 frame=load_data(a.data);X=frame[FEATURES];y=frame[TARGET]
 Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
 models={
 "logistic_regression":pipe(LogisticRegression(max_iter=3000,class_weight="balanced",solver="liblinear")),
 "random_forest":pipe(RandomForestClassifier(n_estimators=250,min_samples_leaf=2,class_weight="balanced_subsample",random_state=42,n_jobs=-1)),
 "dummy_baseline":pipe(DummyClassifier(strategy="prior"))}
 cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42); cvout={}
 for name,m in models.items():
  s=cross_validate(m,Xtr,ytr,cv=cv,scoring={"ap":"average_precision","auc":"roc_auc","f1":"f1","recall":"recall"},n_jobs=1)
  cvout[name]={k:{"mean":float(v.mean()),"std":float(v.std())} for k,v in {"average_precision":s["test_ap"],"roc_auc":s["test_auc"],"f1":s["test_f1"],"recall":s["test_recall"]}.items()}
 best=max(["logistic_regression","random_forest"],key=lambda n:cvout[n]["average_precision"]["mean"])
 model=models[best].fit(Xtr,ytr); prob=model.predict_proba(Xte)[:,1]; pred=(prob>=.5).astype(int)
 report=classification_report(yte,pred,target_names=["no_failure","failure"],output_dict=True,zero_division=0)
 metrics={"dataset":{"name":"AI4I 2020 Predictive Maintenance Dataset","source":"UCI Machine Learning Repository, dataset 601","doi":"10.24432/C5HS5C","source_note":"UCI identifies this as synthetic data reflecting industrial predictive-maintenance patterns.","rows":len(frame),"features":FEATURES,"target":TARGET,"failure_rate":float(y.mean()),"train_rows":len(Xtr),"test_rows":len(Xte),"split":"Stratified 80/20 holdout, random_state=42"},"selected_model":best,"holdout_test":{"accuracy":float(accuracy_score(yte,pred)),"precision":float(precision_score(yte,pred,zero_division=0)),"recall":float(recall_score(yte,pred,zero_division=0)),"f1":float(f1_score(yte,pred,zero_division=0)),"roc_auc":float(roc_auc_score(yte,prob)),"average_precision":float(average_precision_score(yte,prob)),"confusion_matrix":confusion_matrix(yte,pred).tolist(),"classification_report":report},"five_fold_cross_validation_on_training_split":cvout,"threshold":.5,"limitations":["AI4I is synthetic, not real factory telemetry.","Random split does not test future-time or unseen-machine generalization.","Failure classes are imbalanced; review recall and average precision."]}
 out=ROOT/"outputs";out.mkdir(exist_ok=True);joblib.dump(model,out/"maintenance_model.joblib")
 (out/"metrics.json").write_text(json.dumps(metrics,indent=2));(out/"classification_report.json").write_text(json.dumps(report,indent=2))
 pd.DataFrame({"actual_failure":yte.to_numpy(),"failure_probability":prob,"predicted_failure_at_0_5":pred}).to_csv(out/"holdout_predictions.csv",index=False)
 ConfusionMatrixDisplay.from_predictions(yte,pred,display_labels=["No failure","Failure"],cmap="Blues");plt.title("AI4I 2020 — holdout confusion matrix");plt.tight_layout();plt.savefig(out/"confusion_matrix.png",dpi=160);plt.close()
 RocCurveDisplay.from_predictions(yte,prob);plt.title("AI4I 2020 — ROC curve");plt.tight_layout();plt.savefig(out/"roc_curve.png",dpi=160);plt.close()
 PrecisionRecallDisplay.from_predictions(yte,prob);plt.title("AI4I 2020 — precision-recall curve");plt.tight_layout();plt.savefig(out/"precision_recall_curve.png",dpi=160);plt.close()
 clf=model.named_steps["classifier"];names=model.named_steps["preprocess"].get_feature_names_out();imp=clf.feature_importances_ if hasattr(clf,"feature_importances_") else np.abs(clf.coef_[0]);order=np.argsort(imp)
 plt.figure(figsize=(8,5));plt.barh(np.asarray(names)[order],np.asarray(imp)[order]);plt.xlabel("Model importance");plt.title(f"Feature importance — {best}");plt.tight_layout();plt.savefig(out/"feature_importance.png",dpi=160);plt.close()
 print(json.dumps({"selected_model":best,"holdout_test":metrics["holdout_test"],"cross_validation":cvout},indent=2))
if __name__=="__main__":main()
