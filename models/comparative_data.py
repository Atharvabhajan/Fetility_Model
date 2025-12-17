import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    accuracy_score,
    f1_score,
    auc
)

# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = "model_outputs"

MODEL_FILES = {
    "Logistic Regression": f"{BASE_DIR}/logisticregression_outputs/test_predictions.csv",
    "KNN": f"{BASE_DIR}/knn_outputs/test_predictions_knn.csv",
    "Random Forest": f"{BASE_DIR}/random_forest_strict/test_predictions.csv",
    "XGBoost": f"{BASE_DIR}/xgboost_outputs/test_predictions_xgb.csv"
}

OUTPUT_DIR = "comparative_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD ALL MODEL OUTPUTS
# =====================================================

model_outputs = {}

for model_name, file_path in MODEL_FILES.items():
    df = pd.read_csv(file_path)

    # Standard columns (as you confirmed)
    y_true = df["actual"]
    y_proba = df["probability"]

    model_outputs[model_name] = {
        "y_true": y_true,
        "y_proba": y_proba
    }

print("✅ All model prediction files loaded successfully")

# =====================================================
# ROC CURVE COMPARISON (ALL MODELS IN ONE IMAGE)
# =====================================================

plt.figure(figsize=(8, 6))

for model_name, data in model_outputs.items():
    fpr, tpr, _ = roc_curve(data["y_true"], data["y_proba"])
    auc_score = roc_auc_score(data["y_true"], data["y_proba"])

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{model_name} (AUC = {auc_score:.3f})"
    )

plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison – All Models")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_curve_comparison.png", dpi=300)
plt.show()

# =====================================================
# PRECISION–RECALL CURVE COMPARISON
# =====================================================

plt.figure(figsize=(8, 6))

for model_name, data in model_outputs.items():
    precision, recall, _ = precision_recall_curve(
        data["y_true"], data["y_proba"]
    )
    pr_auc = auc(recall, precision)

    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=f"{model_name} (AUC = {pr_auc:.3f})"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/pr_curve_comparison.png", dpi=300)
plt.show()

# =====================================================
# ACCURACY / F1 / ROC-AUC BAR COMPARISON
# =====================================================

rows = []

for model_name, data in model_outputs.items():
    y_pred = (data["y_proba"] >= 0.5).astype(int)

    rows.append({
        "Model": model_name,
        "Accuracy": accuracy_score(data["y_true"], y_pred),
        "F1 Score": f1_score(data["y_true"], y_pred),
        "ROC-AUC": roc_auc_score(data["y_true"], data["y_proba"])
    })

summary_df = pd.DataFrame(rows)

summary_df.set_index("Model").plot(
    kind="bar",
    figsize=(9, 6)
)

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_comparison_bar.png", dpi=300)
plt.show()

summary_df.to_csv(
    f"{OUTPUT_DIR}/model_comparison_summary.csv",
    index=False
)

print("\n📊 FINAL COMPARISON SUMMARY\n")
print(summary_df)
