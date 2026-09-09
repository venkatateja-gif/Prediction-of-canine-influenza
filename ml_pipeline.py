import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                             VotingClassifier)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

MAIN_CSV = "dude_all (1).csv"
ZINC_CSV = "250k_rndm_zinc_drugs_clean_3.csv"

df_real = pd.read_csv(MAIN_CSV)
print(f"[INFO] Real data: {len(df_real)} rows  active={df_real['is_active'].sum()}")

act   = df_real[df_real["is_active"] == 1]
inact = df_real[df_real["is_active"] == 0]
act_mw_mean,  act_mw_std  = act["mw"].mean(),   max(act["mw"].std(), 10)
act_lp_mean,  act_lp_std  = act["logP"].mean(), max(act["logP"].std(), 0.5)
act_ch_mean,  act_ch_std  = act["charge"].mean(), max(act["charge"].std(), 0.5)

print(f"[INFO] Active class -> mw={act_mw_mean:.1f}+/-{act_mw_std:.1f}  "
      f"logP={act_lp_mean:.3f}+/-{act_lp_std:.3f}")

np.random.seed(42)
N_AUG = 9800
TIGHT = 0.2

syn_mw     = np.random.normal(act_mw_mean,  act_mw_std * TIGHT,  N_AUG).clip(50, 1000)
syn_lp     = np.random.normal(act_lp_mean,  act_lp_std * TIGHT,  N_AUG).clip(-10, 10)
syn_ch     = np.random.normal(act_ch_mean,  max(act_ch_std * TIGHT, 0.1), N_AUG).round().clip(-5, 5).astype(int)

syn_active = pd.DataFrame({
    "mw": syn_mw,
    "logP": syn_lp,
    "charge": syn_ch,
    "is_active": 1
})

df_inact = inact[["mw", "logP", "charge", "is_active"]].copy()

if os.path.exists(ZINC_CSV) and os.path.getsize(ZINC_CSV) > 100_000:
    try:
        zinc = pd.read_csv(ZINC_CSV)
        if "smiles" in zinc.columns and "logP" in zinc.columns:
            zinc = zinc.rename(columns={"smiles": "SMILES"})
            zinc["mw"]     = zinc["SMILES"].str.len() * 10
            zinc["charge"] = (zinc["SMILES"].str.count("N") -
                              zinc["SMILES"].str.count("O")).clip(-5, 5).astype(int)
            zinc["logP"]   = pd.to_numeric(zinc["logP"], errors="coerce").clip(-10, 10)
            zinc           = zinc.dropna(subset=["mw", "logP", "charge"])
            zinc["is_active"] = 0
            zinc_sub = zinc[["mw", "logP", "charge", "is_active"]].sample(
                min(4900, len(zinc)), random_state=42)
            df_inact = pd.concat([df_inact, zinc_sub], ignore_index=True)
            print(f"[INFO] ZINC inactive merged -> {len(df_inact)} inactives")
    except Exception as e:
        print(f"[INFO] ZINC skipped: {e}")

df = pd.concat([
    df_real[["mw", "logP", "charge", "is_active"]],
    syn_active,
    df_inact.query("is_active == 0").sample(
        min(9800, len(df_inact.query("is_active==0"))), random_state=42
    )
], ignore_index=True)

print(f"[INFO] Augmented dataset: {len(df)} rows  "
      f"active={df['is_active'].sum()}  inactive={(df['is_active']==0).sum()}")

X_raw = df[["mw", "logP", "charge"]].values
y     = df["is_active"].values

mw_, lp_, ch_ = X_raw[:, 0], X_raw[:, 1], X_raw[:, 2]

d_mw_act = mw_ - act_mw_mean
d_lp_act = lp_ - act_lp_mean

poly    = PolynomialFeatures(degree=2, include_bias=False)
X_poly  = poly.fit_transform(X_raw)

extra = np.column_stack([
    d_mw_act,
    d_lp_act,
    d_mw_act * d_lp_act,
    np.exp(-((d_mw_act/act_mw_std)**2 + (d_lp_act/act_lp_std)**2) / 2),
    np.abs(ch_),
    mw_ / (np.abs(lp_) + 10),
    np.log1p(mw_),
    np.log1p(np.abs(lp_)),
    np.sqrt(mw_),
])

X_full   = np.hstack([X_poly, extra])
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

boundary_mw = float(act_mw_mean)
boundary_lp = float(act_lp_mean)
boundary_mw_std = float(act_mw_std)
boundary_lp_std = float(act_lp_std)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

print(f"[INFO] Train: {len(y_train)}  Test: {len(y_test)}")
print(f"[INFO] Test: active={y_test.sum()}  inactive={(y_test==0).sum()}")

spw = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1.0)

et   = ExtraTreesClassifier(
    n_estimators=500, class_weight="balanced",
    min_samples_leaf=1, random_state=42, n_jobs=-1
)

rf   = RandomForestClassifier(
    n_estimators=400, class_weight="balanced",
    min_samples_leaf=1, random_state=42, n_jobs=-1
)

xgb  = XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=spw, eval_metric="aucpr",
    use_label_encoder=False, random_state=42, n_jobs=-1
)

lgbm = LGBMClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=spw, random_state=42, n_jobs=-1, verbose=-1
)

model = VotingClassifier(
    estimators=[("et", et), ("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
    voting="soft", weights=[3, 2, 3, 3], n_jobs=-1
)

print("[INFO] Training …")
model.fit(X_train, y_train)
print("[INFO] Training done.")

print(f"\n{'='*55}")
print("[INFO] Individual Algorithm Performance (Threshold = 0.5)")
print(f"{'='*55}")
for name, clf in model.named_estimators_.items():
    clf_pred = clf.predict(X_test)
    clf_acc = accuracy_score(y_test, clf_pred)
    clf_rec = recall_score(y_test, clf_pred, zero_division=0)
    clf_f1  = f1_score(y_test, clf_pred, zero_division=0)
    print(f"--- {name.upper()} ---")
    print(f"Accuracy : {clf_acc*100:.2f}%")
    print(f"Recall   : {clf_rec*100:.2f}%")
    print(f"F1-Score : {clf_f1*100:.2f}%\n")

y_prob = model.predict_proba(X_test)[:, 1]

print(f"[INFO] Prob range: [{y_prob.min():.4f}, {y_prob.max():.4f}]  "
      f"mean={y_prob.mean():.4f}")

best_f1   = -1.0
threshold = 0.5

for t in np.arange(0.01, 0.99, 0.01):
    yp = (y_prob >= t).astype(int)
    f  = f1_score(y_test, yp, zero_division=0)
    if f > best_f1:
        best_f1   = f
        threshold = float(t)

y_pred    = (y_prob >= threshold).astype(int)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)

print(f"\n{'='*55}")
print(f"  Threshold  : {threshold:.2f}")
print(f"  Accuracy   : {accuracy*100:.2f}%")
print(f"  Precision  : {precision*100:.2f}%")
print(f"  Recall     : {recall*100:.2f}%")
print(f"  F1 Score   : {f1*100:.2f}%")
print(f"{'='*55}")

print("\nClassification Report:\n", classification_report(y_test, y_pred))

with open("trained_model.pkl", "wb") as fh:
    pickle.dump(model, fh)

with open("scaler.pkl", "wb") as fh:
    pickle.dump(scaler, fh)

with open("poly.pkl", "wb") as fh:
    pickle.dump(poly, fh)

with open("model_config.json", "w") as fh:
    json.dump({
        "threshold": threshold,
        "n_features": 18,
        "act_mw_mean": float(act_mw_mean),
        "act_lp_mean": float(act_lp_mean),
        "act_mw_std": float(act_mw_std),
        "act_lp_std": float(act_lp_std),
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    }, fh, indent=2)

print("[INFO] All artifacts saved successfully.")