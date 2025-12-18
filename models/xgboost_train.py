import pickle
import pandas as pd

# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = "./model_outputs/xgboost_outputs/xgb_conception_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

EXPECTED_FEATURES = model.get_booster().feature_names

print("\nXGBoost model loaded successfully ✅")

# ============================================================
# USER INPUT (ONLY REALISTIC INPUTS)
# ============================================================
print("\nEnter patient details:\n")

AGE = float(input("AGE: "))
NO_OF_TIMES_CONCEIVED = float(input("NO_OF_TIMES_CONCEIVED: "))
BORN_ALIVE_TOTAL = float(input("BORN_ALIVE_TOTAL: "))
AGE_AT_FIRST_CONCEPTION = float(input("AGE_AT_FIRST_CONCEPTION: "))

CHEW = int(input("CHEW (0=No, 1=Yes): "))
SMOKE = int(input("SMOKE (0=No, 1=Yes): "))
ALCOHOL = int(input("ALCOHOL (0=No, 1=Yes): "))

RURAL = int(input("RURAL (1=Rural, 0=Urban): "))
DISABILITY_STATUS = int(input("DISABILITY_STATUS (0=No, 1=Yes): "))

# ============================================================
# FEATURE ENGINEERING (MATCH TRAINING)
# ============================================================
age_squared = AGE ** 2
years_since_first_birth = max(AGE - AGE_AT_FIRST_CONCEPTION, 0)

live_birth_success_rate = (
    BORN_ALIVE_TOTAL / NO_OF_TIMES_CONCEIVED
    if NO_OF_TIMES_CONCEIVED > 0 else 0
)

age_x_parity = AGE * NO_OF_TIMES_CONCEIVED
substance_use_score = CHEW + SMOKE + ALCOHOL
has_any_disability = DISABILITY_STATUS

# ============================================================
# BASE FEATURE DICTIONARY (ALL TRAIN FEATURES)
# ============================================================
input_data = {
    "AGE": AGE,
    "NO_OF_TIMES_CONCEIVED": NO_OF_TIMES_CONCEIVED,
    "BORN_ALIVE_TOTAL": BORN_ALIVE_TOTAL,
    "AGE_AT_FIRST_CONCEPTION": AGE_AT_FIRST_CONCEPTION,
    "MOTHER_AGE_WHEN_BABY_WAS_BORN": AGE_AT_FIRST_CONCEPTION,
    "DIAGNOSED_FOR": 0,
    "SYMPTOMS_PERTAINING_ILLNESS": 0,
    "DISABILITY_STATUS": DISABILITY_STATUS,
    "CHEW": CHEW,
    "SMOKE": SMOKE,
    "ALCOHOL": ALCOHOL,
    "HIGHEST_QUALIFICATION": 0,
    "OCCUPATION_STATUS": 0,
    "SOCIAL_GROUP_CODE": 0,
    "RURAL": RURAL,
    "TOILET_USED": 0,
    "COOKING_FUEL": 0,
    "DRINKING_WATER_SOURCE": 0,
    "age_squared": age_squared,
    "years_since_first_birth": years_since_first_birth,
    "live_birth_success_rate": live_birth_success_rate,
    "age_x_parity": age_x_parity,
    "has_any_disability": has_any_disability,
    "substance_use_score": substance_use_score,
    "chronic_disease_count": 0,
    "age_x_chronic_disease": AGE * 0,
    "is_literate": 1,
    "works_outside_home": 0,
    "uses_clean_fuel": 0,
    "practices_open_defecation": 0,
    "safe_drinking_water": 1,
    "environmental_risk_score": RURAL
}

# ============================================================
# ENSURE EXACT FEATURE ORDER
# ============================================================
input_df = pd.DataFrame([input_data])
input_df = input_df[EXPECTED_FEATURES]

# ============================================================
# PREDICTION
# ============================================================
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

# ============================================================
# OUTPUT
# ============================================================
print("\n================= RESULT =================")
print("Prediction  :", "CAN CONCEIVE ✅" if prediction == 1 else "CANNOT CONCEIVE ❌")
print(f"Probability : {probability:.4f}")
print("==========================================")
