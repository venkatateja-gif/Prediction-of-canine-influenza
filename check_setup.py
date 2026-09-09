
import streamlit
import pandas
import numpy
import pickle
import os

print("Imports successful.")

if os.path.exists("trained_models.pkl"):
    try:
        with open("trained_models.pkl", "rb") as f:
            models = pickle.load(f)
        print("trained_models.pkl loaded.")
        if "best_model" in models and "best_threshold" in models:
            print("Keys found in trained_models.pkl")
        else:
            print("Keys MISSING in trained_models.pkl: ", models.keys())
    except Exception as e:
        print(f"Error loading trained_models.pkl: {e}")
else:
    print("trained_models.pkl not found.")

if os.path.exists("scaler.pkl"):
    try:
        with open("scaler.pkl", "rb") as f:
            pickle.load(f)
        print("scaler.pkl loaded.")
    except Exception as e:
        print(f"Error loading scaler.pkl: {e}")
else:
    print("scaler.pkl not found.")
