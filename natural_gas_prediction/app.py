from flask import Flask, render_template, request
import numpy as np
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# Set the correct paths
model_path = 'RandomForestRegressor.pkl'  # Adjust if needed
scaler_path = 'scaler.pkl'  # Assuming you have a scaler saved similarly

# Load the trained model and scaler using pickle
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    year = int(request.form['year'])
    month = int(request.form['month'])
    day = int(request.form['day'])

    # Create input array
    input_data = np.array([[day, month, year]])

    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Make prediction
    price = model.predict(scaled_input)[0]

    # Round the price to 2 decimal places
    price = round(price, 2)

    return render_template('home.html', price=price)

if __name__ == '__main__':
    app.run(debug=True)
