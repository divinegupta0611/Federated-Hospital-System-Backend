from django.shortcuts import render
import json
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import xgboost as xgb
import os
from django.conf import settings

# ============== DIABETES MODEL ==============
MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "prediction",
    "ml",
    "diabetes_xgboost_recall_optimized.json"
)

# Load XGBoost model
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

@csrf_exempt
def predict_diabetes(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=400)

    try:
        data = json.loads(request.body)

        # Extract input values
        gender = 1 if data["gender"] == "Male" else 0
        age = float(data["age"])
        hypertension = int(data["hypertension"])
        heart_disease = int(data["heart_disease"])
        smoking_history = {
            "never": 0,
            "former": 1,
            "current": 2,
            "No Info": 3,
        }.get(data["smoking_history"], 3)

        bmi = float(data["bmi"])
        hba1c = float(data["hba1c_level"])
        glucose = float(data["blood_glucose_level"])

        # Prepare input array
        features = np.array([[gender, age, hypertension, heart_disease,
                              smoking_history, bmi, hba1c, glucose]])

        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])

        return JsonResponse({
            "prediction": prediction,
            "probability": probability
        })

    except Exception as e:
        import traceback
        print("DIABETES ERROR:", traceback.format_exc())
        return JsonResponse({"error": str(e)}, status=500)


# ============== HEART DISEASE MODEL ==============
HEART_MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "prediction",
    "ml",
    "heart_xgboost.json"
)

# Load model
heart_model = xgb.XGBClassifier()
heart_model.load_model(HEART_MODEL_PATH)

@csrf_exempt
def predict_heart_disease(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=400)

    try:
        data = json.loads(request.body)

        # Extract all input fields
        features = np.array([[
            float(data["age"]),
            int(data["sex"]),
            int(data["cp"]),
            float(data["trestbps"]),
            float(data["chol"]),
            int(data["fbs"]),
            int(data["restecg"]),
            float(data["thalach"]),
            int(data["exang"]),
            float(data["oldpeak"]),
            int(data["slope"]),
            int(data["ca"]),
            int(data["thal"])
        ]])

        prediction = int(heart_model.predict(features)[0])
        probability = float(heart_model.predict_proba(features)[0][1])

        return JsonResponse({
            "prediction": prediction,
            "probability": probability
        })

    except Exception as e:
        import traceback
        print("HEART ERROR:", traceback.format_exc())
        return JsonResponse({"error": str(e)}, status=500)
    

# ============== KIDNEY DISEASE MODEL ==============
KIDNEY_MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "prediction",
    "ml",
    "kidney_xgboost.json"
)

# Load Kidney XGBoost model
kidney_model = xgb.XGBClassifier()
kidney_model.load_model(KIDNEY_MODEL_PATH)

print(f"✅ Kidney model loaded from: {KIDNEY_MODEL_PATH}")
print(f"✅ Model file exists: {os.path.exists(KIDNEY_MODEL_PATH)}")

@csrf_exempt
def predict_kidney_disease(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=400)

    try:
        data = json.loads(request.body)
        
        # Print received data for debugging
        print("📩 Received data:", data)
        
        # ✅ FIX ERROR 2: Map string values to integers
        map_binary = {
            "normal": 1,
            "abnormal": 0,
            "present": 1,
            "notpresent": 0,
            "yes": 1,
            "no": 0,
            "good": 0,
            "poor": 1
        }
        
        # Convert categorical string values to integers
        rbc = map_binary.get(data["rbc"].lower(), 0) if isinstance(data["rbc"], str) else int(data["rbc"])
        pc = map_binary.get(data["pc"].lower(), 0) if isinstance(data["pc"], str) else int(data["pc"])
        pcc = map_binary.get(data["pcc"].lower(), 0) if isinstance(data["pcc"], str) else int(data["pcc"])
        ba = map_binary.get(data["ba"].lower(), 0) if isinstance(data["ba"], str) else int(data["ba"])
        htn = map_binary.get(data["htn"].lower(), 0) if isinstance(data["htn"], str) else int(data["htn"])
        dm = map_binary.get(data["dm"].lower(), 0) if isinstance(data["dm"], str) else int(data["dm"])
        cad = map_binary.get(data["cad"].lower(), 0) if isinstance(data["cad"], str) else int(data["cad"])
        appet = map_binary.get(data["appet"].lower(), 0) if isinstance(data["appet"], str) else int(data["appet"])
        pe = map_binary.get(data["pe"].lower(), 0) if isinstance(data["pe"], str) else int(data["pe"])
        ane = map_binary.get(data["ane"].lower(), 0) if isinstance(data["ane"], str) else int(data["ane"])
        
        # ✅ FIX ERROR 1: Add AGE as the FIRST feature (24 features total)
        # Dataset column order: age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane
        features = np.array([[ 
            float(data["age"]),       # 1. AGE (THIS WAS MISSING!)
            float(data["bp"]),        # 2. Blood Pressure
            float(data["sg"]),        # 3. Specific Gravity
            int(data["al"]),          # 4. Albumin
            int(data["su"]),          # 5. Sugar
            rbc,                      # 6. Red Blood Cells (converted)
            pc,                       # 7. Pus Cell (converted)
            pcc,                      # 8. Pus Cell Clumps (converted)
            ba,                       # 9. Bacteria (converted)
            float(data["bgr"]),       # 10. Blood Glucose Random
            float(data["bu"]),        # 11. Blood Urea
            float(data["sc"]),        # 12. Serum Creatinine
            float(data["sod"]),       # 13. Sodium
            float(data["pot"]),       # 14. Potassium
            float(data["hemo"]),      # 15. Hemoglobin
            float(data["pcv"]),       # 16. Packed Cell Volume
            float(data["wc"]),        # 17. White Blood Cell Count
            float(data["rc"]),        # 18. Red Blood Cell Count
            htn,                      # 19. Hypertension (converted)
            dm,                       # 20. Diabetes Mellitus (converted)
            cad,                      # 21. Coronary Artery Disease (converted)
            appet,                    # 22. Appetite (converted)
            pe,                       # 23. Pedal Edema (converted)
            ane                       # 24. Anemia (converted)
        ]])
        
        print("✅ Features array shape:", features.shape)
        print("✅ Features array:", features)
        
        prediction = int(kidney_model.predict(features)[0])
        probability = float(kidney_model.predict_proba(features)[0][1])

        print(f"✅ Prediction: {prediction}, Probability: {probability}")

        return JsonResponse({
            "prediction": prediction,
            "probability": probability
        })

    except KeyError as e:
        import traceback
        error_msg = f"Missing required field: {str(e)}"
        print("❌ KeyError:", error_msg)
        print(traceback.format_exc())
        return JsonResponse({"error": error_msg}, status=400)
    except Exception as e:
        # ✅ FIX ERROR 3: Print full traceback to see real error
        import traceback
        error_msg = str(e)
        print("❌ KIDNEY ERROR:", error_msg)
        print(traceback.format_exc())
        return JsonResponse({"error": error_msg}, status=500)
    
# ============== BREAST CANCER MODEL ==============

BREAST_CANCER_MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "prediction",
    "ml",
    "breast_cancer_model.json"
)

# Load Breast Cancer XGBoost model
breast_model = xgb.XGBClassifier()
breast_model.load_model(BREAST_CANCER_MODEL_PATH)

print(f"✅ Breast Cancer model loaded from: {BREAST_CANCER_MODEL_PATH}")
print(f"✅ Model file exists: {os.path.exists(BREAST_CANCER_MODEL_PATH)}")


@csrf_exempt
def predict_cancer(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=400)

    try:
        data = json.loads(request.body)

        # ===== Extract Features in Proper Order =====
        mean_radius = float(data["mean_radius"])
        mean_texture = float(data["mean_texture"])
        mean_perimeter = float(data["mean_perimeter"])
        mean_area = float(data["mean_area"])
        mean_smoothness = float(data["mean_smoothness"])

        # ===== Prepare input array =====
        features = np.array([[ 
            mean_radius,
            mean_texture,
            mean_perimeter,
            mean_area,
            mean_smoothness
        ]])

        prediction = int(breast_model.predict(features)[0])
        probability = float(breast_model.predict_proba(features)[0][1])

        return JsonResponse({
            "prediction": prediction,
            "probability": probability
        })

    except KeyError as e:
        error_msg = f"Missing required field: {str(e)}"
        print("❌ KeyError:", error_msg)
        return JsonResponse({"error": error_msg}, status=400)

    except Exception as e:
        import traceback
        print("❌ BREAST CANCER ERROR:", str(e))
        print(traceback.format_exc())
        return JsonResponse({"error": str(e)}, status=500)
    

# ============== PARKINSON DISEASE MODEL ==============

PARKINSON_MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "prediction",
    "ml",
    "parkinson_model.json"
)

# Load Parkinson XGBoost model
parkinson_model = xgb.XGBClassifier()
parkinson_model.load_model(PARKINSON_MODEL_PATH)

print(f"✅ Parkinson model loaded from: {PARKINSON_MODEL_PATH}")
print(f"✅ Model file exists: {os.path.exists(PARKINSON_MODEL_PATH)}")

@csrf_exempt
def predict_parkinson(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=400)

    try:
        data = json.loads(request.body)

        # Change these to match frontend's snake_case keys
        features = np.array([[ 
            float(data["mdvp_fo"]),              # Changed
            float(data["mdvp_fhi"]),             # Changed
            float(data["mdvp_flo"]),             # Changed
            float(data["mdvp_jitter_percent"]),  # Changed
            float(data["mdvp_jitter_abs"]),      # Changed
            float(data["mdvp_rap"]),             # Changed
            float(data["mdvp_ppq"]),             # Changed
            float(data["jitter_ddp"]),           # Changed
            float(data["mdvp_shimmer"]),         # Changed
            float(data["mdvp_shimmer_db"]),      # Changed
            float(data["shimmer_apq3"]),         # Changed
            float(data["shimmer_apq5"]),         # Changed
            float(data["mdvp_apq"]),             # Changed
            float(data["shimmer_dda"]),          # Changed
            float(data["nhr"]),                  # Changed
            float(data["hnr"]),                  # Changed
            float(data["rpde"]),                 # Changed
            float(data["dfa"]),                  # Changed
            float(data["spread1"]),              # Changed
            float(data["spread2"]),              # Changed
            float(data["d2"]),                   # Changed
            float(data["ppe"])                   # Changed
        ]])

        prediction = int(parkinson_model.predict(features)[0])
        probability = float(parkinson_model.predict_proba(features)[0][1])

        return JsonResponse({
            "prediction": prediction,
            "probability": probability
        })

    except KeyError as e:
        error_msg = f"Missing required field: {str(e)}"
        print("❌ KeyError:", error_msg)
        return JsonResponse({"error": error_msg}, status=400)

    except Exception as e:
        import traceback
        print("❌ PARKINSON ERROR:", str(e))
        print(traceback.format_exc())
        return JsonResponse({"error": str(e)}, status=500)