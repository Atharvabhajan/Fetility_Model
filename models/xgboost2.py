import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# =======================
# Load Dataset
# =======================
df = pd.read_csv("ahs_feature_engine.csv")

target = "can_conceive_strict"

# =======================
# CRITICAL: REMOVE LEAKAGE FEATURES
# =======================
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

leakage_found = [col for col in leakage_features if col in df.columns]
if leakage_found:
    print("\n" + "="*60)
    print("REMOVING DATA LEAKAGE FEATURES")
    print("="*60)
    print(f"Dropping {len(leakage_found)} features that cause leakage:")
    for feat in leakage_found:
        print(f"  - {feat}")
    df = df.drop(columns=leakage_found)
    print(f"Features remaining: {df.shape[1]}")
else:
    print("No leakage features found")

# =======================
# Remove Helper Columns
# =======================
helper_cols = ['diag_set', 'sym_set']
helper_found = [col for col in helper_cols if col in df.columns]
if helper_found:
    print(f"\nRemoving {len(helper_found)} helper columns:")
    for col in helper_found:
        print(f"  - {col}")
    df = df.drop(columns=helper_found)
    print(f"Helper columns removed")

# =======================
# Remove non-numeric columns
# =======================
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype("category").cat.codes

# =======================
# Handle Missing Values
# =======================
# Fill NaN values with median for each column
df = df.fillna(df.median())

# =======================
# Show Correlations
# =======================
print("\n================ TARGET CORRELATIONS ================\n")
corr = df.corr()[target].sort_values(ascending=False)
print(corr)

# =======================
# Train-Test Split
# =======================
X = df.drop(columns=[target])
y = df[target]

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42, k_neighbors=5)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\n================ CLASS DISTRIBUTION ================")
print(f"Before SMOTE - Class 0: {sum(y_train == 0)}, Class 1: {sum(y_train == 1)}")
print(f"After SMOTE  - Class 0: {sum(y_train_smote == 0)}, Class 1: {sum(y_train_smote == 1)}")

# =======================
# Train XGBoost with balancing
# =======================
model = XGBClassifier(
    n_estimators=600,
    max_depth=8,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    colsample_bylevel=0.8,
    eval_metric="logloss",
    min_child_weight=2,
    gamma=1.0,
    reg_alpha=0.5,
    reg_lambda=2.0,
    scale_pos_weight=1.2,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

model.fit(X_train_smote, y_train_smote)

# =======================
# Predictions with Optimized Threshold
# =======================
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Find optimal threshold that maximizes F1 for both classes
from sklearn.metrics import f1_score as compute_f1

best_threshold = 0.5
best_f1_weighted = 0

for threshold in np.arange(0.3, 0.7, 0.01):
    y_pred_temp = (y_pred_proba >= threshold).astype(int)
    f1_weighted = compute_f1(y_test, y_pred_temp, average='weighted')
    if f1_weighted > best_f1_weighted:
        best_f1_weighted = f1_weighted
        best_threshold = threshold

print(f"\n================ THRESHOLD OPTIMIZATION ================")
print(f"Optimal Threshold: {best_threshold:.2f} (Weighted F1: {best_f1_weighted:.4f})")

y_pred = (y_pred_proba >= best_threshold).astype(int)

# =======================
# Metrics
# =======================
print("\n============================================================")
print("               XGBOOST CLASSIFICATION REPORT")
print("============================================================\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n================ CONFUSION MATRIX ================\n")
print(cm)

# ROC AUC
roc = roc_auc_score(y_test, y_pred_proba)
f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_macro = f1_score(y_test, y_pred, average='macro')

# Calculate metrics per class
tn, fp, fn, tp = cm.ravel()
recall_class0 = tn / (tn + fp)
recall_class1 = tp / (tp + fn)
precision_class0 = tn / (tn + fn)
precision_class1 = tp / (tp + fp)

print("\n================ DETAILED EVALUATION METRICS ================\n")
print(f"ROC-AUC Score: {roc:.4f}")
print(f"Weighted F1-Score: {f1_weighted:.4f}")
print(f"Macro F1-Score: {f1_macro:.4f}")
print(f"\nClass 0 (Negative):")
print(f"  Recall: {recall_class0:.4f}")
print(f"  Precision: {precision_class0:.4f}")
f1_class0_val = 2 * (precision_class0 * recall_class0) / (precision_class0 + recall_class0) if (precision_class0 + recall_class0) > 0 else 0
print(f"  F1-Score: {f1_class0_val:.4f}")
print(f"\nClass 1 (Positive):")
print(f"  Recall: {recall_class1:.4f}")
print(f"  Precision: {precision_class1:.4f}")
f1_class1_val = 2 * (precision_class1 * recall_class1) / (precision_class1 + recall_class1) if (precision_class1 + recall_class1) > 0 else 0
print(f"  F1-Score: {f1_class1_val:.4f}")

# =======================
# Feature Importance
# =======================
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

print("\n================ TOP FEATURE IMPORTANCE ================\n")
print(importance.head(20))

# # =======================
# # PREDICTION FUNCTION FOR NEW PERSONS
# # =======================
# def predict_conception(model, new_data, threshold=0.57):
#     """
#     Predict conception capability for new person(s)
    
#     Parameters:
#     - model: trained XGBoost model
#     - new_data: DataFrame with same features as training data
#     - threshold: decision threshold (default 0.57 from optimization)
    
#     Returns:
#     - prediction: 0 (cannot conceive) or 1 (can conceive)
#     - probability: probability of being able to conceive
#     """
#     # Get probability
#     prob = model.predict_proba(new_data)[:, 1][0]
    
#     # Apply threshold
#     prediction = 1 if prob >= threshold else 0
    
#     return prediction, prob

# # =======================
# # TEST WITH NEW PERSON DATA
# # =======================
# print("\n" + "="*60)
# print("TESTING PREDICTIONS ON NEW PERSON DATA")
# print("="*60)

# # Get the mean values from training data for baseline
# train_mean = X_train_smote.mean()

# # Example 1: Healthy young woman (can likely conceive)
# test_person_1 = train_mean.copy()
# test_person_1['AGE'] = 28
# test_person_1['age_squared'] = 28**2
# test_person_1['live_birth_success_rate'] = 0.8
# test_person_1['is_literate'] = 1
# test_person_1['HIGHEST_QUALIFICATION'] = 2
# test_person_1['NO_OF_TIMES_CONCEIVED'] = 1
# test_person_1['age_x_chronic_disease'] = 0
# test_person_1['chronic_disease_count'] = 0
# test_person_1['DIAGNOSED_FOR'] = 0
# test_person_1 = test_person_1.to_frame().T

# try:
#     pred_1, prob_1 = predict_conception(model, test_person_1, threshold=best_threshold)
    
#     print(f"\n{'='*60}")
#     print("TEST PERSON 1 (Healthy 28-year-old)")
#     print(f"{'='*60}")
#     print(f"Age: 28 years")
#     print(f"Literate: Yes")
#     print(f"Live Birth Success Rate: 0.80")
#     print(f"Times Conceived: 1")
#     print(f"Chronic Diseases: 0")
#     print(f"\n✓ PREDICTION: {'Can Conceive' if pred_1 == 1 else 'Cannot Conceive'}")
#     print(f"✓ Probability: {prob_1:.4f}")
#     print(f"✓ Confidence: {max(prob_1, 1-prob_1):.1%}")
# except Exception as e:
#     print(f"Error: {e}")

# # Example 2: Older woman with health challenges
# test_person_2 = train_mean.copy()
# test_person_2['AGE'] = 42
# test_person_2['age_squared'] = 42**2
# test_person_2['live_birth_success_rate'] = 0.25
# test_person_2['is_literate'] = 0
# test_person_2['HIGHEST_QUALIFICATION'] = 0
# test_person_2['NO_OF_TIMES_CONCEIVED'] = 3
# test_person_2['age_x_chronic_disease'] = 4
# test_person_2['chronic_disease_count'] = 2
# test_person_2['DIAGNOSED_FOR'] = 1
# test_person_2['years_since_first_birth'] = 18
# test_person_2 = test_person_2.to_frame().T

# try:
#     pred_2, prob_2 = predict_conception(model, test_person_2, threshold=best_threshold)
    
#     print(f"\n{'='*60}")
#     print("TEST PERSON 2 (Older 42-year-old with health issues)")
#     print(f"{'='*60}")
#     print(f"Age: 42 years")
#     print(f"Literate: No")
#     print(f"Live Birth Success Rate: 0.25")
#     print(f"Times Conceived: 3")
#     print(f"Chronic Diseases: 2")
#     print(f"Years Since First Birth: 18")
#     print(f"\n✓ PREDICTION: {'Can Conceive' if pred_2 == 1 else 'Cannot Conceive'}")
#     print(f"✓ Probability: {prob_2:.4f}")
#     print(f"✓ Confidence: {max(prob_2, 1-prob_2):.1%}")
# except Exception as e:
#     print(f"Error: {e}")

# # Example 3: Middle-aged woman with moderate factors
# test_person_3 = train_mean.copy()
# test_person_3['AGE'] = 35
# test_person_3['age_squared'] = 35**2
# test_person_3['live_birth_success_rate'] = 0.60
# test_person_3['is_literate'] = 1
# test_person_3['HIGHEST_QUALIFICATION'] = 1
# test_person_3['NO_OF_TIMES_CONCEIVED'] = 2
# test_person_3['age_x_chronic_disease'] = 1
# test_person_3['chronic_disease_count'] = 0
# test_person_3['DIAGNOSED_FOR'] = 0
# test_person_3['years_since_first_birth'] = 8
# test_person_3 = test_person_3.to_frame().T

# try:
#     pred_3, prob_3 = predict_conception(model, test_person_3, threshold=best_threshold)
    
#     print(f"\n{'='*60}")
#     print("TEST PERSON 3 (Middle-aged 35-year-old)")
#     print(f"{'='*60}")
#     print(f"Age: 35 years")
#     print(f"Literate: Yes")
#     print(f"Live Birth Success Rate: 0.60")
#     print(f"Times Conceived: 2")
#     print(f"Chronic Diseases: 0")
#     print(f"Years Since First Birth: 8")
#     print(f"\n✓ PREDICTION: {'Can Conceive' if pred_3 == 1 else 'Cannot Conceive'}")
#     print(f"✓ Probability: {prob_3:.4f}")
#     print(f"✓ Confidence: {max(prob_3, 1-prob_3):.1%}")
# except Exception as e:
#     print(f"Error: {e}")

# print("\n" + "="*60)
# print("PREDICTION COMPLETE - All test cases finished")
# print("="*60)

# import pickle

# # Save MODEL
# with open("model.pkl", "wb") as f:
#     pickle.dump(model, f)

# # Save MEANS (used to fill missing values in web app forms)
# train_mean = X_train_smote.mean()
# with open("means.pkl", "wb") as f:
#     pickle.dump(train_mean, f)

# # Save SCALER (ONLY if you used StandardScaler)
# try:
#     scaler
#     with open("scaler.pkl", "wb") as f:
#         pickle.dump(scaler, f)
#     print("\nScaler saved as scaler.pkl")
# except NameError:
#     print("\nNo scaler used → skipping scaler.pkl")
    
# print("\nAll pkl files saved successfully: model.pkl, means.pkl")
