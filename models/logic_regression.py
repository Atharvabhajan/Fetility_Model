import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

# ============================================================
# 1. LOAD DATA
# ============================================================

FILE_PATH = "./dataset/final_model_ready.csv"   # change path
MODEL_NAME = "LogisticRegression"

df = pd.read_csv(FILE_PATH)
print("Dataset shape:", df.shape)

# ============================================================
# 2. REMOVE LEAKAGE FEATURES
# ============================================================

leakage_features = [
    'IS_CURRENTLY_PREGNANT',
    'IS_CURRENTLY_MENSTRUATING',
    'IS_TUBECTOMY',
    'has_repro_disease',
    'has_repro_disease_diag',
    'has_repro_disease_sym',
    '_direct_repro_flag'
]

TARGET = 'can_conceive_strict'

df = df.drop(columns=leakage_features, errors='ignore')

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("Target distribution:\n", y.value_counts(normalize=True))

# ============================================================
# 3. TRAIN–TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# 4. IMPUTATION
# ============================================================

imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# ============================================================
# 5. SCALING (IMPORTANT FOR LOGISTIC REGRESSION)
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 6. MODEL TRAINING
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    solver='lbfgs'
)

model.fit(X_train_scaled, y_train)

# ============================================================
# 7. PREDICTIONS
# ============================================================

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

# ============================================================
# 8. METRICS
# ============================================================

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, y_test_proba)

print("\n========== MODEL PERFORMANCE ==========")
print(f"Train Accuracy : {train_acc:.4f}")
print(f"Test Accuracy  : {test_acc:.4f}")
print(f"ROC-AUC Score  : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'{MODEL_NAME} - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_logreg.png', dpi=300)
plt.show()

# ============================================================
# 10. ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(y_test, y_test_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'{MODEL_NAME} - ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve_logreg.png', dpi=300)
plt.show()

# ============================================================
# 11. PRECISION–RECALL CURVE
# ============================================================

precision, recall, _ = precision_recall_curve(y_test, y_test_proba)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'{MODEL_NAME} - Precision Recall Curve')
plt.tight_layout()
plt.savefig('pr_curve_logreg.png', dpi=300)
plt.show()

# ============================================================
# 12. FEATURE IMPORTANCE (COEFFICIENTS)
# ============================================================

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

feature_importance['AbsCoeff'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values(
    by='AbsCoeff', ascending=False
)

feature_importance.to_csv('feature_importance_logreg.csv', index=False)

# ============================================================
# 13. SAVE MODEL & SCALER
# ============================================================

joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(scaler, 'scaler.pkl')

# ============================================================
# 14. SAVE PERFORMANCE SUMMARY (FOR COMPARISON TABLE)
# ============================================================

summary = pd.DataFrame([{
    'Model': MODEL_NAME,
    'Train Accuracy': train_acc,
    'Test Accuracy': test_acc,
    'ROC_AUC': roc_auc
}])

summary.to_csv('model_performance_logreg.csv', index=False)

print("\nAll artifacts saved successfully ✅")
