import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib


data = pd.read_csv("insuranceFraud_Dataset.csv")


data.columns = data.columns.str.strip()
print(data.columns)  

data['policy_bind_date'] = pd.to_datetime(data['policy_bind_date'], errors='coerce')
data['incident_date'] = pd.to_datetime(data['incident_date'], errors='coerce')


data['policy_bind_date'] = (data['policy_bind_date'] - data['policy_bind_date'].min()).dt.days
data['incident_date'] = (data['incident_date'] - data['incident_date'].min()).dt.days


data = pd.get_dummies(data, drop_first=True)


print(data.columns) 

X = data.drop("fraud_reported_Y", axis=1) 
y = data['fraud_reported_Y']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)


joblib.dump(model, 'fraud_detection_model.pkl')
