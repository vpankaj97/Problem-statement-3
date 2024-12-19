from flask import Flask, request, jsonify
import joblib
import numpy as np
import pickle

# Load the trained model
model = joblib.load('fraud_detection_model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.json
    input_data = np.array([data['features']])
    prediction = model.predict(input_data)[0]
    return jsonify({'Fraudulent': bool(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
