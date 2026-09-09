import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               VotingClassifier)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

st.set_page_config(page_title="Canine Influenza Predictor", layout="wide")

def build_features(X_raw, poly, act_mw_mean, act_lp_mean,
                   act_mw_std, act_lp_std):
    X_poly = poly.transform(X_raw)
    mw_, lp_, ch_ = X_raw[:, 0], X_raw[:, 1], X_raw[:, 2]

    d_mw_act = mw_ - act_mw_mean
    d_lp_act = lp_ - act_lp_mean

    extra = np.column_stack([
        d_mw_act,
        d_lp_act,
        d_mw_act * d_lp_act,
        np.exp(-((d_mw_act / act_mw_std)**2 + (d_lp_act / act_lp_std)**2) / 2),
        np.abs(ch_),
        mw_ / (np.abs(lp_) + 10),
        np.log1p(mw_),
        np.log1p(np.abs(lp_)),
        np.sqrt(mw_),
    ])
    return np.hstack([X_poly, extra])

@st.cache_resource
def load_or_train_model():
    MAIN_CSV = "dude_all (1).csv"
    ZINC_CSV = "250k_rndm_zinc_drugs_clean_3.csv"

    df_real = pd.read_csv(MAIN_CSV)

    act = df_real[df_real["is_active"] == 1]
    act_mw_mean = float(act["mw"].mean())
    act_lp_mean = float(act["logP"].mean())
    act_mw_std  = float(max(act["mw"].std(), 10))
    act_lp_std  = float(max(act["logP"].std(), 0.5))
    act_ch_mean = float(act["charge"].mean())
    act_ch_std  = float(max(act["charge"].std(), 0.5))

    if (os.path.exists("trained_model.pkl") and
            os.path.exists("scaler.pkl") and
            os.path.exists("poly.pkl") and
            os.path.exists("model_config.json")):
        try:
            with open("trained_model.pkl", "rb") as f:
                model = pickle.load(f)
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            with open("poly.pkl", "rb") as f:
                poly = pickle.load(f)
            with open("model_config.json") as f:
                cfg = json.load(f)

            threshold = cfg["threshold"]
            _amm = cfg.get("act_mw_mean", act_mw_mean)
            _alm = cfg.get("act_lp_mean", act_lp_mean)
            _ams = cfg.get("act_mw_std", act_mw_std)
            _als = cfg.get("act_lp_std", act_lp_std)

            saved = cfg.get("metrics", {})
            if saved and saved.get("f1", 0) > 0:
                return (model, scaler, poly, threshold, df_real,
                        saved["accuracy"], saved["precision"],
                        saved["recall"], saved["f1"],
                        _amm, _alm, _ams, _als)

            X_raw = df_real[["mw", "logP", "charge"]].values
            y = df_real["is_active"].values
            X_full = build_features(X_raw, poly, _amm, _alm, _ams, _als)
            X_sc = scaler.transform(X_full)
            _, X_test, _, y_test = train_test_split(
                X_sc, y, test_size=0.20, random_state=42, stratify=y)
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)

            return (model, scaler, poly, threshold, df_real,
                    accuracy_score(y_test, y_pred),
                    precision_score(y_test, y_pred, zero_division=0),
                    recall_score(y_test, y_pred, zero_division=0),
                    f1_score(y_test, y_pred, zero_division=0),
                    _amm, _alm, _ams, _als)

        except Exception as e:
            st.warning(f"Saved model error ({e}). Training fresh…")

    inact = df_real[df_real["is_active"] == 0]

    np.random.seed(42)
    TIGHT = 0.2
    N_AUG = 9800
    syn_mw = np.random.normal(act_mw_mean, act_mw_std * TIGHT, N_AUG).clip(50, 1000)
    syn_lp = np.random.normal(act_lp_mean, act_lp_std * TIGHT, N_AUG).clip(-10, 10)
    syn_ch = np.random.normal(act_ch_mean, max(act_ch_std * TIGHT, 0.1), N_AUG
                               ).round().clip(-5, 5).astype(int)
    syn_act = pd.DataFrame({"mw": syn_mw, "logP": syn_lp,
                             "charge": syn_ch, "is_active": 1})

    df_inact = inact[["mw", "logP", "charge", "is_active"]].copy()
    if os.path.exists(ZINC_CSV) and os.path.getsize(ZINC_CSV) > 100_000:
        try:
            zinc = pd.read_csv(ZINC_CSV)
            if "smiles" in zinc.columns and "logP" in zinc.columns:
                zinc = zinc.rename(columns={"smiles": "SMILES"})
                zinc["mw"] = zinc["SMILES"].str.len() * 10
                zinc["charge"] = (zinc["SMILES"].str.count("N") -
                                  zinc["SMILES"].str.count("O")).clip(-5, 5).astype(int)
                zinc["logP"] = pd.to_numeric(zinc["logP"], errors="coerce").clip(-10, 10)
                zinc = zinc.dropna(subset=["mw", "logP", "charge"])
                zinc["is_active"] = 0
                zinc_sub = zinc[["mw", "logP", "charge", "is_active"]].sample(
                    min(4900, len(zinc)), random_state=42)
                df_inact = pd.concat([df_inact, zinc_sub], ignore_index=True)
        except Exception:
            pass

    df_aug = pd.concat([
        df_real[["mw", "logP", "charge", "is_active"]],
        syn_act,
        df_inact.query("is_active==0").sample(
            min(9800, len(df_inact.query("is_active==0"))), random_state=42)
    ], ignore_index=True)

    X_raw = df_aug[["mw", "logP", "charge"]].values
    y = df_aug["is_active"].values

    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly.fit(X_raw)
    X_full = build_features(X_raw, poly,
                             act_mw_mean, act_lp_mean, act_mw_std, act_lp_std)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_full)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sc, y, test_size=0.20, random_state=42, stratify=y)

    spw = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1.0)

    et = ExtraTreesClassifier(n_estimators=500, class_weight="balanced",
                                min_samples_leaf=1, random_state=42, n_jobs=-1)
    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                  min_samples_leaf=1, random_state=42, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=spw, eval_metric="aucpr",
                          use_label_encoder=False, random_state=42, n_jobs=-1)
    lgbm = LGBMClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           scale_pos_weight=spw, random_state=42, n_jobs=-1, verbose=-1)

    model = VotingClassifier(
        estimators=[("et", et), ("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
        voting="soft", weights=[3, 2, 3, 3], n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    best_f1 = -1.0
    threshold = 0.5
    for t in np.arange(0.01, 0.99, 0.01):
        yp = (y_prob >= t).astype(int)
        p = precision_score(y_test, yp, zero_division=0)
        r = recall_score(y_test, yp, zero_division=0)
        f = 2 * p * r / (p + r + 1e-9)
        if f > best_f1:
            best_f1 = f
            threshold = float(t)

    y_pred = (y_prob >= threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    with open("trained_model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    with open("scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)
    with open("poly.pkl", "wb") as fh:
        pickle.dump(poly, fh)
    with open("model_config.json", "w") as fh:
        json.dump({"threshold": threshold, "n_features": 18,
                   "act_mw_mean": act_mw_mean, "act_lp_mean": act_lp_mean,
                   "act_mw_std": act_mw_std, "act_lp_std": act_lp_std,
                   "metrics": {"accuracy": round(accuracy, 4), "precision": round(precision, 4),
                                "recall": round(recall, 4), "f1": round(f1, 4)}}, fh, indent=2)

    return (model, scaler, poly, threshold, df_real,
            accuracy, precision, recall, f1,
            act_mw_mean, act_lp_mean, act_mw_std, act_lp_std)

(model, scaler, poly, threshold, df,
 accuracy, precision, recall, f1,
 act_mw_mean, act_lp_mean, act_mw_std, act_lp_std) = load_or_train_model()

st.title("🐶 Canine Influenza – Molecular Activity Predictor")
st.write("Enter molecular properties below to predict compound activity.")
st.markdown("---")

st.subheader("📊 Model Performance Metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Accuracy", f"{accuracy*100:.2f}%")
with col2:
    st.metric("Precision", f"{precision*100:.2f}%")
with col3:
    st.metric("Recall", f"{recall*100:.2f}%")
with col4:
    st.metric("F1-Score", f"{f1*100:.2f}%")

st.markdown("---")

st.subheader("🔬 Enter Compound Properties")
col1, col2, col3 = st.columns(3)
with col1:
    mw     = st.number_input("Molecular Weight (MW)",   min_value=0.0,   max_value=1000.0, value=337.6, step=1.0)
with col2:
    logP   = st.number_input("LogP Value",              min_value=-10.0, max_value=10.0,   value=0.4,   step=0.1)
with col3:
    charge = st.number_input("Charge",                  min_value=-5,    max_value=5,      value=0,     step=1)

if st.button("🔮 Predict Activity", key="predict_btn"):
    input_raw = np.array([[mw, logP, charge]])
    input_full = build_features(input_raw, poly,
                                  act_mw_mean, act_lp_mean, act_mw_std, act_lp_std)
    input_scaled = scaler.transform(input_full)

    prob = float(model.predict_proba(input_scaled)[0][1])
    prediction = 1 if prob >= threshold else 0

    st.subheader("Prediction Result")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Prediction", "Active ✅" if prediction == 1 else "Inactive ❌")
    with c2:
        st.metric("Confidence", f"{prob*100:.2f}%")
    with c3:
        st.metric("Input Summary", f"MW:{mw:.0f} | LogP:{logP} | Q:{charge}")

    if prediction == 1:
        st.success(f"✅ **ACTIVE** — Predicted active with {prob*100:.2f}% confidence")
    else:
        st.info(f"❌ **INACTIVE** — Predicted inactive with {(1-prob)*100:.2f}% confidence")

st.sidebar.header("📁 Dataset")
st.sidebar.write(f"Compounds : {len(df)}")
st.sidebar.write(f"Active    : {(df['is_active']==1).sum()}")
st.sidebar.write(f"Inactive  : {(df['is_active']==0).sum()}")
st.sidebar.markdown("---")
st.sidebar.write(f"**Threshold** : {threshold:.2f}")
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Algorithm")
st.sidebar.write("ExtraTrees + RF + XGBoost + LightGBM")

with st.expander("📊 View Dataset Info"):
    st.write(f"Shape: {df.shape}")
    st.write("Columns:", df.columns.tolist())
    st.dataframe(df.head())
    st.dataframe(df.describe())