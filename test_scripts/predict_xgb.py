import joblib
import pandas as pd

# ==============================
# LOAD MODEL
# ==============================

MODEL_PATH = "./model_outputs/xgboost_outputs/xgb_conception_model.pkl"

model = joblib.load(MODEL_PATH)
print("✅ XGBoost model loaded successfully")


# ==============================
# PREDICTION FUNCTION
# ==============================

def predict_conception(user_input: dict):
    """
    user_input: dictionary of feature_name -> value
    """

    df = pd.DataFrame([user_input])

    # Predict
    probability = model.predict_proba(df)[0][1]
    prediction = int(probability >= 0.5)

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "label": "Can Conceive" if prediction == 1 else "Cannot Conceive"
    }


# ==============================
# CLI USER INPUT
# ==============================

if __name__ == "__main__":

    print("\nEnter patient details:\n")

    user_data = {
        "Age": float(input("Age: ")),
        "BMI": float(input("BMI: ")),
        "Hemoglobin": float(input("Hemoglobin: ")),
        "Cycle_Length": float(input("Cycle Length: ")),
        "TSH": float(input("TSH Level: ")),
        "Prolactin": float(input("Prolactin Level: ")),
        "FSH": float(input("FSH Level: ")),
        "LH": float(input("LH Level: "))
    }

    result = predict_conception(user_data)

    print("\n========== RESULT ==========")
    print(f"Prediction  : {result['label']}")
    print(f"Probability : {result['probability']}")
