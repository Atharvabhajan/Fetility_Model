import pickle
import pandas as pd
import numpy as np

# ================= LOAD MODEL =================
MODEL_PATH = "./model_outputs/xgboost_outputs/xgb_conception_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("\nXGBoost model loaded successfully ✅")

# Get expected feature order directly from model
expected_features = model.get_booster().feature_names

# ================= USER INPUT (BASE FEATURES ONLY) =================
print("\nEnter patient details:\n")

AGE = int(input("AGE: "))
NO_OF_TIMES_CONCEIVED = int(input("NO_OF_TIMES_CONCEIVED: "))
BORN_ALIVE_TOTAL = int(input("BORN_ALIVE_TOTAL: "))
AGE_AT_FIRST_CONCEPTION = int(input("AGE_AT_FIRST_CONCEPTION: "))

SYMPTOMS_PERTAINING_ILLNESS = int(input("SYMPTOMS_PERTAINING_ILLNESS (count): "))
DIAGNOSED_FOR = int(input("DIAGNOSED_FOR (count): "))

CHEW = int(input("CHEW (0=No, 1=Yes): "))
SMOKE = int(input("SMOKE (0=No, 1=Yes): "))
ALCOHOL = int(input("ALCOHOL (0=No, 1=Yes): "))

RURAL = int(input("RURAL (1=Rural, 0=Urban): "))
DISABILITY_STATUS = int(input("DISABILITY_STATUS (0=No, 1=Yes): "))

OCCUPATION_STATUS = int(input("OCCUPATION_STATUS (0=Not Working, 1=Working): "))

# ================= FEATURE ENGINEERING (MATCH TRAINING) =================
age_squared = AGE ** 2

years_since_first_birth = (
    AGE - AGE_AT_FIRST_CONCEPTION if BORN_ALIVE_TOTAL > 0 else 0
)

live_birth_success_rate = (
    BORN_ALIVE_TOTAL / NO_OF_TIMES_CONCEIVED
    if NO_OF_TIMES_CONCEIVED > 0 else 0
)

age_x_parity = AGE * NO_OF_TIMES_CONCEIVED

has_any_disability = DISABILITY_STATUS

substance_use_score = CHEW + SMOKE + ALCOHOL

chronic_disease_count = SYMPTOMS_PERTAINING_ILLNESS + DIAGNOSED_FOR

age_x_chronic_disease = AGE * chronic_disease_count

# ================= FULL FEATURE DICTIONARY =================
feature_dict = {
    "AGE": AGE,
    "NO_OF_TIMES_CONCEIVED": NO_OF_TIMES_CONCEIVED,
    "BORN_ALIVE_TOTAL": BORN_ALIVE_TOTAL,
    "AGE_AT_FIRST_CONCEPTION": AGE_AT_FIRST_CONCEPTION,
    "SYMPTOMS_PERTAINING_ILLNESS": SYMPTOMS_PERTAINING_ILLNESS,
    "DIAGNOSED_FOR": DIAGNOSED_FOR,
    "CHEW": CHEW,
    "SMOKE": SMOKE,
    "ALCOHOL": ALCOHOL,
    "RURAL": RURAL,
    "DISABILITY_STATUS": DISABILITY_STATUS,
    "OCCUPATION_STATUS": OCCUPATION_STATUS,
    "age_squared": age_squared,
    "years_since_first_birth": years_since_first_birth,
    "live_birth_success_rate": live_birth_success_rate,
    "age_x_parity": age_x_parity,
    "has_any_disability": has_any_disability,
    "substance_use_score": substance_use_score,
    "chronic_disease_count": chronic_disease_count,
    "age_x_chronic_disease": age_x_chronic_disease,
}

# Fill missing expected features with 0 (safe)
final_input = {feat: feature_dict.get(feat, 0) for feat in expected_features}

input_df = pd.DataFrame([final_input])

# ================= PREDICTION =================
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

# ================= OUTPUT =================
print("\n================= RESULT =================")
if prediction == 1:
    print("Prediction  : CAN CONCEIVE ✅")
else:
    print("Prediction  : CANNOT CONCEIVE ❌")

print(f"Probability : {probability:.4f}")
print("==========================================")
