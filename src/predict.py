from __future__ import annotations
import argparse
from pathlib import Path
import joblib,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
FEATURES=["Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]","Type"]
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output",default="outputs/new_predictions.csv");a=p.parse_args()
 model_path=ROOT/"outputs"/"maintenance_model.joblib"
 if not model_path.exists():raise FileNotFoundError("Train first with python src/train.py")
 f=pd.read_csv(a.input);missing=sorted(set(FEATURES)-set(f.columns))
 if missing:raise ValueError(f"Missing required columns: {missing}")
 prob=joblib.load(model_path).predict_proba(f[FEATURES])[:,1]
 out=pd.DataFrame({"failure_probability":prob,"predicted_failure_at_0_5":(prob>=.5).astype(int)})
 dest=Path(a.output);dest=dest if dest.is_absolute() else ROOT/dest;dest.parent.mkdir(parents=True,exist_ok=True);out.to_csv(dest,index=False);print(out.head())
if __name__=="__main__":main()
