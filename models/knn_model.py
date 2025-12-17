import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ============================================================
# VISUALIZATION SETTINGS
# ============================================================
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# ============================================================
# OUTPUT DIRECTORY (SAME AS RF & XGB)
# ============================================================
OUTPUT_DIR = "./model_outputs/knn_outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
df = pd.read_csv("./dataset/final_model_ready.csv")

TARGET_COL = "can_conceive_broad"

print("Dataset Shape:", df.shape)
print("\nTarget Distribution:")
print(df[TARGET_COL].value_counts())

# ============================================================
# STEP 2: REMOVE DATA LEAKAGE FEATURES
# ============================================================
leakage_features = [
    'IS_CURRENTLY_PREGNANT',
    'IS_CURRENTLY_MENSTRUATING',
    'IS_TUBECTOMY',
    '_direct_repro_flag',
    'can_conceive_strict'
]

leakage_found = [c for c in leakage_features if c in df.columns]
if leakage_found:
    print("\nREMOVING DATA LEAKAGE FEATURES:")
    for c in leakage_found:
        print(" -", c)
    df.drop(columns=leakage_found, inplace=True)

# ============================================================
# STEP 3: HANDLE NON-NUMERIC & MISSING VALUES
# ============================================================
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype("category").cat.codes

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# ============================================================
# STEP 4: TRAIN–TEST SPLIT (CONSISTENT FORMAT)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# ============================================================
# STEP 5: TRAIN KNN MODEL
# ============================================================
knn = KNeighborsClassifier(n_neighbors=10)

print("\nTRAINING KNN MODEL...")
knn.fit(X_train, y_train)

# ============================================================
# STEP 6: PREDICTIONS (THRESHOLD = 0.5 NOT APPLICABLE)
# ============================================================
y_pred = knn.predict(X_test)

# KNN probability (needed for ROC / PR)
y_pred_proba = knn.predict_proba(X_test)[:, 1]

# ============================================================
# STEP 7: MODEL PERFORMANCE (SAME METRICS)
# ============================================================
print("\n--- KNN MODEL PERFORMANCE ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ============================================================
# STEP 8: CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - KNN")
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_knn.png", dpi=300)
plt.close()

# ============================================================
# STEP 9: ROC CURVE
# ============================================================
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_pred_proba):.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("ROC Curve - KNN")
plt.savefig(f"{OUTPUT_DIR}/roc_curve_knn.png", dpi=300)
plt.close()

# ============================================================
# STEP 10: PRECISION–RECALL CURVE
# ============================================================
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - KNN")
plt.savefig(f"{OUTPUT_DIR}/precision_recall_curve_knn.png", dpi=300)
plt.close()

# ============================================================
# STEP 11: CROSS-VALIDATION (ROC-AUC)
# ============================================================
cv_scores = cross_val_score(
    knn,
    X_train,
    y_train,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

# ============================================================
# STEP 12: SAVE MODEL
# ============================================================
with open(f"{OUTPUT_DIR}/knn_conception_model.pkl", "wb") as f:
    pickle.dump(knn, f)

# ============================================================
# STEP 13: SAVE TEST PREDICTIONS
# ============================================================
results_df = pd.DataFrame({
    "actual": y_test.values,
    "predicted": y_pred,
    "probability": y_pred_proba
})

results_df.to_csv(
    f"{OUTPUT_DIR}/test_predictions_knn.csv",
    index=False
)

# ============================================================
# STEP 14: PERFORMANCE SUMMARY (SAME FORMAT)
# ============================================================
performance_summary = {
    "Model": ["KNN"],
    "Accuracy": [accuracy_score(y_test, y_pred)],
    "F1_Score": [f1_score(y_test, y_pred)],
    "ROC_AUC": [roc_auc_score(y_test, y_pred_proba)],
    "CV_Mean_AUC": [cv_scores.mean()],
    "CV_Std_AUC": [cv_scores.std()]
}

pd.DataFrame(performance_summary).to_csv(
    f"{OUTPUT_DIR}/model_performance_summary_knn.csv",
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n====================================================")
print("KNN TRAINING COMPLETE")
print("====================================================")
print("Outputs saved in:", OUTPUT_DIR)
