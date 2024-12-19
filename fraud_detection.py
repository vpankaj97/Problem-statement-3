import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load sample data
# You can replace this with an actual dataset or a publicly available dataset
url = r"C:\Users\rosha\Downloads\insuranceFraud_Dataset.csv"
data = pd.read_csv(url)
print(data)
# Display data overview
print("Data Overview:")
print(data.head())

# Preprocessing: Drop unnecessary columns
data = data.drop(columns=['policy_number', 'insured_zip', 'incident_location'])

# Encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Split into features (X) and target (y)
X = data.drop(columns=['fraud_reported_Y'])
y = data['fraud_reported_Y']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
