import pickle
import pandas as pd
import numpy as np

# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = "./model_outputs/xgboost_outputs/xgb_conception_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("\nXGBoost model loaded successfully ✅")

# ============================================================
# INPUT VALIDATION FUNCTIONS
# ============================================================
def validate_inputs(data):
    if data["AGE"] < 12 or data["AGE"] > 55:
        raise ValueError("Age must be between 12 and 55")

    if data["AGE_AT_FIRST_CONCEPTION"] > data["AGE"]:
        raise ValueError("Age at first conception cannot exceed current age")

    if data["NO_OF_TIMES_CONCEIVED"] > data["AGE"]:
        raise ValueError("Invalid conception count")

    if data["BORN_ALIVE_TOTAL"] > data["NO_OF_TIMES_CONCEIVED"]:
        raise ValueError("Born alive count cannot exceed conceptions")

# ============================================================
# USER INPUT
# ============================================================
print("\nEnter patient details:\n")

AGE = int(input("AGE: "))
NO_OF_TIMES_CONCEIVED = int(input("NO_OF_TIMES_CONCEIVED: "))
BORN_ALIVE_TOTAL = int(input("BORN_ALIVE_TOTAL: "))
AGE_AT_FIRST_CONCEPTION = int(input("AGE_AT_FIRST_CONCEPTION: "))

CHEW = int(input("CHEW (0=No, 1=Yes): "))
SMOKE = int(input("SMOKE (0=No, 1=Yes): "))
ALCOHOL = int(input("ALCOHOL (0=No, 1=Yes): "))

RURAL = int(input("RURAL (1=Rural, 0=Urban): "))
DISABILITY_STATUS = int(input("DISABILITY_STATUS (0=No, 1=Yes): "))

# ============================================================
# RAW INPUT DICT
# ============================================================
input_data = {
    "AGE": AGE,
    "NO_OF_TIMES_CONCEIVED": NO_OF_TIMES_CONCEIVED,
    "BORN_ALIVE_TOTAL": BORN_ALIVE_TOTAL,
    "AGE_AT_FIRST_CONCEPTION": AGE_AT_FIRST_CONCEPTION,
    "CHEW": CHEW,
    "SMOKE": SMOKE,
    "ALCOHOL": ALCOHOL,
    "RURAL": RURAL,
    "DISABILITY_STATUS": DISABILITY_STATUS
}

# ============================================================
# VALIDATE INPUTS
# ============================================================
try:
    validate_inputs(input_data)
except ValueError as e:
    print(f"\n❌ INPUT ERROR: {e}")
    exit()

# ============================================================
# FEATURE ENGINEERING (MATCHES TRAINING)
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

# Assumed-safe defaults (documented)
chronic_disease_count = 0
age_x_chronic_disease = AGE * chronic_disease_count
is_literate = 1
works_outside_home = 1
uses_clean_fuel = 1
practices_open_defecation = 0
safe_drinking_water = 1
environmental_risk_score = practices_open_defecation + (1 - safe_drinking_water)

# ============================================================
# FINAL FEATURE VECTOR (ORDER MATTERS!)
# ============================================================
final_input = pd.DataFrame([{
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
    "HIGHEST_QUALIFICATION": 1,
    "OCCUPATION_STATUS": 1,
    "SOCIAL_GROUP_CODE": 1,
    "RURAL": RURAL,
    "TOILET_USED": 1,
    "COOKING_FUEL": 1,
    "DRINKING_WATER_SOURCE": 1,
    "age_squared": age_squared,
    "years_since_first_birth": years_since_first_birth,
    "live_birth_success_rate": live_birth_success_rate,
    "age_x_parity": age_x_parity,
    "has_any_disability": has_any_disability,
    "substance_use_score": substance_use_score,
    "chronic_disease_count": chronic_disease_count,
    "age_x_chronic_disease": age_x_chronic_disease,
    "is_literate": is_literate,
    "works_outside_home": works_outside_home,
    "uses_clean_fuel": uses_clean_fuel,
    "practices_open_defecation": practices_open_defecation,
    "safe_drinking_water": safe_drinking_water,
    "environmental_risk_score": environmental_risk_score
}])

# ============================================================
# MODEL PREDICTION
# ============================================================
probability = model.predict_proba(final_input)[0][1]

# ============================================================
# POST-MODEL CALIBRATION (SAFETY)
# ============================================================
if AGE < 18:
    probability *= 0.6

if substance_use_score >= 2:
    probability *= 0.7

probability = round(probability, 4)

# ============================================================
# RISK BAND OUTPUT
# ============================================================
if probability >= 0.75:
    label = "HIGH likelihood"
elif probability >= 0.55:
    label = "MODERATE likelihood"
else:
    label = "LOW likelihood"

# ============================================================
# FINAL OUTPUT
# ============================================================
print("\n================= RESULT =================")
print(f"Likelihood : {label}")
print(f"Probability: {probability}")
print("==========================================\n")
