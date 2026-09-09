import pandas as pd
import numpy as np

df = pd.read_csv("dude_all (1).csv")
print("Columns:", df.columns.tolist())
print("is_active:", df["is_active"].value_counts().to_dict())
print(f"mw median={df['mw'].median():.2f}  logP median={df['logP'].median():.4f}")
synthetic = ((df['mw'] > df['mw'].median()) & (df['logP'] > df['logP'].median())).astype(int)
print(f"Synthetic active={synthetic.sum()}  real active={df['is_active'].sum()}")
print(f"Match rate: {(synthetic == df['is_active']).mean()*100:.1f}%")

act = df[df['is_active']==1]
inact = df[df['is_active']==0]
print(f"\nACTIVE (n={len(act)}):")
print(act[['mw','logP','charge']].agg(['min','max','mean','std']).round(3))
print(f"\nINACTIVE (n={len(inact)}):")
print(inact[['mw','logP','charge']].agg(['min','max','mean','std']).round(3))
