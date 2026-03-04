# ==========================================
# HEART DISEASE PREDICTION - FRAMINGHAM
# ==========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ------------------------------
# STEP 1: Load Dataset
# ------------------------------

df = pd.read_csv("framingham.csv")

print("Dataset Loaded Successfully")
print("Total Rows:", df.shape[0])


# ------------------------------
# STEP 2: Handle Missing Values
# ------------------------------

df = df.dropna()   # remove missing values


# ------------------------------
# STEP 3: Split Features & Target
# ------------------------------

X = df.drop("TenYearCHD", axis=1)
y = df["TenYearCHD"]


# ------------------------------
# STEP 4: Scaling
# ------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ------------------------------
# STEP 5: Train-Test Split
# ------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42)


# ------------------------------
# STEP 6: Train Model
# ------------------------------

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nModel Training Completed")
print("Model Accuracy:", accuracy_score(y_test, model.predict(X_test)))


# ==========================================
# STEP 7: USER INPUT
# ==========================================

print("\nEnter Patient Details:")

# IMPORTANT: These names must match dataset columns exactly
age = float(input("Enter Age: "))
sex = int(input("Enter Sex (0=Female, 1=Male): "))
cigsPerDay = float(input("Enter Cigarettes per Day: "))
totChol = float(input("Enter Cholesterol Level: "))
sysBP = float(input("Enter Systolic BP: "))
diaBP = float(input("Enter Diastolic BP: "))
BMI = float(input("Enter BMI: "))
heartRate = float(input("Enter Heart Rate: "))
glucose = float(input("Enter Glucose Level: "))

# For remaining features (default safe values)
education = 1
currentSmoker = 1 if cigsPerDay > 0 else 0
BPMeds = 0
prevalentStroke = 0
prevalentHyp = 0
diabetes = 0


new_patient = pd.DataFrame({
    "male": [sex],
    "age": [age],
    "education": [education],
    "currentSmoker": [currentSmoker],
    "cigsPerDay": [cigsPerDay],
    "BPMeds": [BPMeds],
    "prevalentStroke": [prevalentStroke],
    "prevalentHyp": [prevalentHyp],
    "diabetes": [diabetes],
    "totChol": [totChol],
    "sysBP": [sysBP],
    "diaBP": [diaBP],
    "BMI": [BMI],
    "heartRate": [heartRate],
    "glucose": [glucose]
})

# Scale input
new_patient_scaled = scaler.transform(new_patient)

# Predict
prediction = model.predict(new_patient_scaled)
probability = model.predict_proba(new_patient_scaled)

print("\n--- RESULT ---")

if prediction[0] == 1:
    print("⚠ High Risk of Heart Disease (10 Year Risk)")
else:
    print("✅ Low Risk of Heart Disease")

print("Probability [No Disease, Disease]:", probability)
