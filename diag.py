import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("dude_all (1).csv")
mw_med = 317.36
lp_med = 0.7053499999999996

X_raw = df[["mw", "logP", "charge"]].values
y     = df["is_active"].values

mw_, lp_, ch_ = X_raw[:, 0], X_raw[:, 1], X_raw[:, 2]
mw_dist  = mw_ - mw_med
lp_dist  = lp_ - lp_med
both_pos = (mw_dist > 0).astype(float) * (lp_dist > 0).astype(float)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_raw)
extra = np.column_stack([
    mw_dist, lp_dist, both_pos, mw_dist * lp_dist,
    np.log1p(np.abs(mw_dist)), np.log1p(np.abs(lp_dist)),
    np.abs(ch_), mw_ / (np.abs(lp_) + 10), np.sqrt(np.abs(mw_)),
])
X_full = np.hstack([X_poly, extra])

scaler = StandardScaler()
X_sc = scaler.fit_transform(X_full)
_, X_test, _, y_test = train_test_split(X_sc, y, test_size=0.20, random_state=42, stratify=y)

with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)

y_prob = model.predict_proba(X_test)[:, 1]

# Save diagnostic to file
with open("diag_output.txt", "w") as out:
    out.write(f"Prob min={y_prob.min():.4f}  max={y_prob.max():.4f}\n")
    out.write(f"Prob mean={y_prob.mean():.4f}  median={np.median(y_prob):.4f}\n")
    out.write(f"pct<0.1: {(y_prob<0.1).mean()*100:.1f}%\n")
    out.write(f"pct<0.3: {(y_prob<0.3).mean()*100:.1f}%\n")
    out.write(f"pct<0.5: {(y_prob<0.5).mean()*100:.1f}%\n")
    out.write(f"pct>0.8: {(y_prob>0.8).mean()*100:.1f}%\n")
    out.write(f"pct>0.9: {(y_prob>0.9).mean()*100:.1f}%\n")
    out.write(f"Test active={y_test.sum()}  inactive={(y_test==0).sum()}\n\n")
    out.write("Threshold scan:\n")
    for t in np.arange(0.05, 0.99, 0.05):
        yp = (y_prob >= t).astype(int)
        p = precision_score(y_test, yp, zero_division=0)
        r = recall_score(y_test, yp, zero_division=0)
        f = f1_score(y_test, yp, zero_division=0)
        out.write(f"  t={t:.2f}  pos={yp.sum():4d}  P={p:.3f}  R={r:.3f}  F1={f:.3f}\n")

print("Done. See diag_output.txt")
