# ==========================================
# CREDIT CARD FRAUD - LOF (FINAL CLEAN VERSION)
# ==========================================

import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------
# STEP 1: Load Dataset
# ------------------------------

df = pd.read_csv("creditcard.csv")

print("Dataset Loaded Successfully")
print("Total rows:", df.shape[0])

# ------------------------------
# STEP 2: Take Small Sample
# ------------------------------

df = df.sample(n=10000, random_state=42)

# ------------------------------
# STEP 3: Prepare Features
# ------------------------------

# Drop Time column
df = df.drop("Time", axis=1)

X = df.drop("Class", axis=1)   # Features
y = df["Class"]                # Target

print("Feature columns:")
print(X.columns)

# ------------------------------
# STEP 4: Scale Data
# ------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# STEP 5: Apply LOF
# ------------------------------

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.02
)

y_pred = lof.fit_predict(X_scaled)

# Convert (-1,1) → (1,0)
y_pred = np.where(y_pred == -1, 1, 0)

# ------------------------------
# STEP 6: Evaluation
# ------------------------------

print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))

print("\nClassification Report:")
print(classification_report(y, y_pred))

print("\nActual Fraud Cases:", sum(y))
print("Detected Fraud Cases:", sum(y_pred))
