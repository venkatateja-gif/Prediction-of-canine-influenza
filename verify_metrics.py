import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load Data
df = pd.read_csv("dude_all (1).csv")
X = df[['mw', 'logP', 'charge']].values
y = df["is_active"]

# Standardize features (must match original pipeline logic, hopefully scaler was fit on X before split in original?)
# In the user's code:
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# X_train, X_test, ... = train_test_split(X, ...)
# This means we must do the same.

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Load Models
with open("trained_models.pkl", "rb") as f:
    models = pickle.load(f)

print(f"Loaded models for: {list(models.keys())}")

results = {}

# Write results to file
with open("verification_report.txt", "w") as f:
    for name, model in models.items():
        if name in ["BestModelName", "BestThreshold"]:
            continue
            
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            
            # Use simple 0.5 threshold first
            preds = (probs >= 0.5).astype(int)
            
            # Also check with BestThreshold if it applies (e.g. to BestModel)
            threshold = 0.5
            if name == models.get("BestModelName"):
                 threshold = models.get("BestThreshold", 0.5)
                 preds_optimized = (probs >= threshold).astype(int)
                 f.write(f"\n--- {name} (Optimized Threshold: {threshold:.4f}) ---\n")
                 f.write(classification_report(y_test, preds_optimized))
                 f.write(f"Recall: {recall_score(y_test, preds_optimized)}\n")
                 f.write(f"F1: {f1_score(y_test, preds_optimized)}\n")
            
            f.write(f"\n--- {name} (Default Threshold: 0.5) ---\n")
            f.write(f"Recall: {recall_score(y_test, preds)}\n")
            f.write(f"F1: {f1_score(y_test, preds)}\n")
            
        else:
            preds = model.predict(X_test)
            f.write(f"\n--- {name} ---\n")
            f.write(f"Recall: {recall_score(y_test, preds)}\n")
            f.write(f"F1: {f1_score(y_test, preds)}\n")

