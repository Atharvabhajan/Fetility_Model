import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_recall_curve,
    f1_score
)

warnings.filterwarnings('ignore')

# ============================================================
# VISUALIZATION SETTINGS
# ============================================================
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================
# OUTPUT DIRECTORY (LOCAL)
# ============================================================
OUTPUT_DIR = "./model_outputs/logisticregression_outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
df = pd.read_csv("./dataset/final_model_ready.csv")

print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nTarget Variable Distribution:")
print(df['can_conceive_strict'].value_counts())

# ============================================================
# STEP 2: FEATURES & TARGET
# ============================================================
TARGET_COL = 'can_conceive_strict'

X = df.drop([TARGET_COL], axis=1)
y = df[TARGET_COL]

# ============================================================
# REMOVE DATA LEAKAGE FEATURES
# ============================================================
leakage_features = [
    'IS_CURRENTLY_PREGNANT',
    'IS_CURRENTLY_MENSTRUATING',
    'IS_TUBECTOMY',
    'has_repro_disease',
    'has_repro_disease_diag',
    'has_repro_disease_sym',
    'can_conceive_broad',
    '_direct_repro_flag'
]

leakage_found = [col for col in leakage_features if col in X.columns]
if leakage_found:
    print("\nREMOVING DATA LEAKAGE FEATURES:")
    for col in leakage_found:
        print(" -", col)
    X = X.drop(columns=leakage_found)

helper_cols = ['diag_set', 'sym_set']
helper_found = [col for col in helper_cols if col in X.columns]
if helper_found:
    X = X.drop(columns=helper_found)

print("\nFinal Feature Shape:", X.shape)
print("Target Shape:", y.shape)

# ============================================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# STEP 4: IMPUTATION & SCALING
# ============================================================
imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# STEP 5: BASELINE LOGISTIC REGRESSION
# ============================================================
logreg = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    solver='lbfgs'
)

logreg.fit(X_train_scaled, y_train)

y_pred = logreg.predict(X_test_scaled)
y_pred_proba = logreg.predict_proba(X_test_scaled)[:, 1]

print("\nLOGISTIC REGRESSION MODEL PERFORMANCE")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
print(classification_report(y_test, y_pred))

# ============================================================
# STEP 6: FEATURE IMPORTANCE (COEFFICIENTS)
# ============================================================
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': logreg.coef_[0],
    'abs_coefficient': np.abs(logreg.coef_[0])
}).sort_values(by='abs_coefficient', ascending=False)

feature_importance.to_csv(
    f"{OUTPUT_DIR}/feature_importance.csv",
    index=False
)

plt.figure(figsize=(10, 8))
top20 = feature_importance.head(20)
plt.barh(top20['feature'], top20['coefficient'])
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Coefficients")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=300)
plt.close()

# ============================================================
# STEP 7: CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=300)
plt.close()

# ============================================================
# STEP 8: ROC CURVE
# ============================================================
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_pred_proba):.3f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", dpi=300)
plt.close()

# ============================================================
# STEP 9: PRECISION-RECALL CURVE
# ============================================================
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.savefig(f"{OUTPUT_DIR}/precision_recall_curve.png", dpi=300)
plt.close()

# ============================================================
# STEP 10: CROSS-VALIDATION
# ============================================================
cv_scores = cross_val_score(
    logreg,
    X_train_scaled,
    y_train,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

print("\nCV Mean ROC-AUC:", cv_scores.mean())

# ============================================================
# STEP 11: SAVE MODEL
# ============================================================
with open(f"{OUTPUT_DIR}/logistic_regression_model.pkl", "wb") as f:
    pickle.dump(logreg, f)

with open(f"{OUTPUT_DIR}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open(f"{OUTPUT_DIR}/imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)

# ============================================================
# STEP 12: SAVE TEST PREDICTIONS
# ============================================================
results_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_pred,
    'probability': y_pred_proba
})

results_df.to_csv(
    f"{OUTPUT_DIR}/test_predictions.csv",
    index=False
)

print("\nTRAINING COMPLETE")
print("All outputs saved in:", OUTPUT_DIR)
