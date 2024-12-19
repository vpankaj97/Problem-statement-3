from flask import Flask, request, jsonify
import joblib
import numpy as np
import pickle


model = joblib.load(r'fraud_detection_model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.json
    input_data = np.array([data['features']])
    prediction = model.predict(input_data)[0]
    return jsonify({'Fraudulent': bool(prediction)})

if __name__ == "__main__":
    app.run(debug=True)

