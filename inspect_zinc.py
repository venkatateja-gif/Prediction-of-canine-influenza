import pandas as pd
import sys

try:
    df = pd.read_csv("250k_rndm_zinc_drugs_clean_3.csv", nrows=5)
    print("Columns:", df.columns.tolist())
    print("Head:\n", df.head(3).to_string())
except Exception as e:
    print("Error:", e)
    try:
        df = pd.read_csv("250k_rndm_zinc_drugs_clean_3.csv", nrows=5, encoding="latin-1")
        print("With latin-1 encoding:")
        print("Columns:", df.columns.tolist())
        print("Head:\n", df.head(3).to_string())
    except Exception as e2:
        print("Error 2:", e2)
