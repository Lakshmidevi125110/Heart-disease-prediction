# ==========================================
# HEART DISEASE PREDICTION USING LOGISTIC REGRESSION
# ==========================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Step 2: Load Dataset (UPDATED PATH)
df = pd.read_csv(r"C:\Users\ELCOT\Downloads\framingham.csv")

print("First 5 Rows:")
print(df.head())

print("\nChecking Missing Values:")
print(df.isnull().sum())


# Step 3: Data Cleaning

# Drop unnecessary column if exists
if 'education' in df.columns:
    df.drop(columns=['education'], inplace=True)

# Rename column (optional)
if 'male' in df.columns:
    df.rename(columns={'male': 'Sex_male'}, inplace=True)

# Remove missing values
df.dropna(inplace=True)

print("\nAfter Cleaning Shape:", df.shape)


# Step 4: Exploratory Data Analysis
plt.figure(figsize=(6,4))
sns.countplot(x='TenYearCHD', data=df)
plt.title("Distribution of Heart Disease")
plt.show()


# Step 5: Select Features and Target
X = np.asarray(df[['age', 'Sex_male', 'cigsPerDay', 
                   'totChol', 'sysBP', 'glucose']])

y = np.asarray(df['TenYearCHD'])


# Step 6: Scale Features
X = preprocessing.StandardScaler().fit(X).transform(X)


# Step 7: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=4)

print("\nTrain set:", X_train.shape)
print("Test set:", X_test.shape)


# Step 8: Train Logistic Regression Model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# Step 9: Make Predictions
y_pred = logreg.predict(X_test)


# Step 10: Evaluate Model

# Accuracy
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

import pandas as pd
conf_matrix = pd.DataFrame(cm,
                           columns=['Predicted:0', 'Predicted:1'],
                           index=['Actual:0', 'Actual:1'])

plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Greens")
plt.title("Confusion Matrix")
plt.show()
