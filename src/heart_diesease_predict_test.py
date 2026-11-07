import pandas as pd
import joblib
from pathlib import Path
 
ROOT = Path(__file__).resolve().parent.parent
MODEL_LOCATION = ROOT / "model"

#model = joblib.load(MODEL_LOCATION/"heart_logReg_model.joblib")
model = joblib.load(MODEL_LOCATION/"heart_tree_model.joblib")
#model = joblib.load(MODEL_LOCATION/"heart_knn_model.joblib")
#model = joblib.load(MODEL_LOCATION/"heart_gnb_model.joblib")
#model = joblib.load(MODEL_LOCATION/"heart_svm_model.joblib")


patient = {
    "Age": 25,
    "Gender": "Male",            
    "Weight": 200,              
    "Height": 183,                 
    "BMI": 35.8,
    "Smoking": "Current",          
    "Alcohol_Intake": "High", 
    "Physical_Activity": "Sedentary",  
    "Diet": "Unhealthy",       
    "Stress_Level": "High",     
    "Hypertension": 0,          
    "Diabetes": 1,
    "Hyperlipidemia": 0,
    "Family_History": 0,
    "Previous_Heart_Attack": 1,
    "Systolic_BP": 120,
    "Diastolic_BP": 80,
    "Heart_Rate": 60,
    "Blood_Sugar_Fasting": 100,
    "Cholesterol_Total": 190,
}

patient_df = pd.DataFrame([patient])

prediction = model.predict(patient_df)[0]
probability = model.predict_proba(patient_df)[0][1]

print(f"Predicted class : {prediction}  (1 = heart disease, 0 = no heart disease)")
print(f"Probability of heart disease: {probability * 100:.1f}%")