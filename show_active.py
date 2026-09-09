import pandas as pd
import json

df = pd.read_csv("dude_all (1).csv")
act = df[df["is_active"] == 1][["mw","logP","charge"]]
print("Active compound stats:")
print(act.describe().round(2))
print("\nSample real active compounds:")
print(act.head(8).to_string())

with open("model_config.json") as f:
    cfg = json.load(f)
print(f"\nModel threshold : {cfg['threshold']}")
print(f"Active centroid : MW={cfg['act_mw_mean']:.1f}  logP={cfg['act_lp_mean']:.3f}")
