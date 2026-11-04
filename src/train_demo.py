"""Train an illustrative bearing-fault demo on generated signals (not JNU measurements)."""
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
CLASSES=["normal","inner_race","outer_race","rolling_element"]
def signal(label,rpm,rng,n=2048,fs=50000):
 t=np.arange(n)/fs; shaft=rpm/60
 f={"normal":2.3,"inner_race":5.4,"outer_race":3.6,"rolling_element":4.7}[label]*shaft
 amp={"normal":.05,"inner_race":.65,"outer_race":.60,"rolling_element":.58}[label]
 x=.15*np.sin(2*np.pi*shaft*t)+amp*np.sin(2*np.pi*f*t+rng.uniform(0,2*np.pi))
 for h in range(2,6): x+=(amp/(h*1.8))*np.sin(2*np.pi*f*h*t+rng.uniform(0,2*np.pi))
 return x+rng.normal(0,.25 if label=="normal" else .32,n)
def features(x):
 x=x-np.mean(x); spec=np.abs(np.fft.rfft(x)); bands=np.array_split(spec,10)
 return [np.std(x),np.sqrt(np.mean(x*x)),np.max(np.abs(x)),np.mean(np.abs(x)),np.mean(x*x),np.percentile(np.abs(x),95),np.max(np.abs(x))/(np.sqrt(np.mean(x*x))+1e-9),np.mean(spec),np.max(spec)]+[np.sum(b*b)/(np.sum(spec*spec)+1e-9) for b in bands]
def main():
 p=argparse.ArgumentParser();p.add_argument("--samples-per-class",type=int,default=500);p.add_argument("--output-dir",default="outputs");a=p.parse_args()
 rng=np.random.default_rng(42);rows=[]
 for label in CLASSES:
  for _ in range(a.samples_per_class):
   rpm=int(rng.choice([600,800,1000]));rows.append(features(signal(label,rpm,rng))+[rpm,label])
 cols=[f"f{i}" for i in range(19)]+["rpm","label"];df=pd.DataFrame(rows,columns=cols)
 X=df.drop(columns="label");y=df.label;Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
 rf=RandomForestClassifier(n_estimators=40,max_depth=8,min_samples_leaf=3,class_weight="balanced",random_state=42,n_jobs=-1)
 svm=make_pipeline(StandardScaler(),SVC(C=4,gamma="scale",class_weight="balanced"))
 rf.fit(Xtr,ytr);svm.fit(Xtr,ytr);pr=rf.predict(Xte);ps=svm.predict(Xte)
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);joblib.dump(rf,out/"bearing_fault_random_forest.joblib",compress=9)
 metrics={"dataset_type":"Generated illustrative vibration signals; NOT measured JNU data","reference_dataset":"https://github.com/ClarkGableWang/JNU-Bearing-Dataset","samples":len(df),"classes":CLASSES,"features":list(X.columns),"seed":42,"train_samples":len(Xtr),"test_samples":len(Xte),"models":{"Random Forest":{"accuracy":float(accuracy_score(yte,pr)),"macro_f1":float(f1_score(yte,pr,average="macro"))},"SVM RBF":{"accuracy":float(accuracy_score(yte,ps)),"macro_f1":float(f1_score(yte,ps,average="macro"))}},"selected_model":"Random Forest","classification_report":classification_report(yte,pr,output_dict=True),"confusion_matrix":confusion_matrix(yte,pr,labels=CLASSES).tolist(),"disclaimer":"Simulation-only metrics. Not a JNU dataset result or evidence of industrial performance."}
 (out/"metrics.json").write_text(json.dumps(metrics,indent=2));pd.DataFrame({"actual":yte,"predicted":pr}).to_csv(out/"test_predictions.csv",index=False)
 print(json.dumps(metrics["models"],indent=2))
if __name__=="__main__":main()
