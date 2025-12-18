# Bar graph for XGBoost feature importance

import pandas as pd
import matplotlib.pyplot as plt
import os

# ================== PATHS ==================
input_path = r"./model_outputs/xgboost_outputs/feature_importance_xgb.csv"
output_path = r"./model_outputs/xgboost_outputs/xgb_feature_importance_bar.png"

# ================== LOAD DATA ==================
df = pd.read_csv(input_path)

# ================== SORT ==================
df = df.sort_values(by="importance", ascending=False)

# ================== PLOT ==================
plt.figure(figsize=(10, 8))
plt.barh(df["feature"], df["importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")

# Highest importance at top
plt.gca().invert_yaxis()

plt.tight_layout()

# ================== SAVE FIGURE ==================
plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()

print(f"✅ Feature importance plot saved at:\n{output_path}")
