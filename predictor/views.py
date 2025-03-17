from django.shortcuts import render
import os
import numpy as np
import pandas as pd
import joblib  # For model saving/loading
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Get the absolute path of the dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'predictor', 'crop_yield_data.csv')

# Load dataset
df = pd.read_csv(DATASET_PATH)
X = df[['Rainfall (mm)', 'Temperature (°C)', 'Fertilizer (kg/hectare)']].values
y = df['Crop Yield (tons/hectare)'].values

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Transforming features to polynomial features
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_scaled)

# Train the model
model = LinearRegression()
model.fit(X_poly, y)

# Calculate R² score
y_pred = model.predict(X_poly)
r2 = r2_score(y, y_pred)
print(f"Model R-squared Score: {r2:.4f}")  # Print R² score in console

# Save the model and scalers
joblib.dump(model, 'predictor/crop_yield_model.pkl')
joblib.dump(scaler, 'predictor/scaler.pkl')
joblib.dump(poly, 'predictor/poly.pkl')

# Prediction function
def predict_yield(request):
    if request.method == 'POST':
        rainfall = float(request.POST['rainfall'])
        temperature = float(request.POST['temperature'])
        fertilizer = float(request.POST['fertilizer'])

        # Load trained model and scalers
        model = joblib.load('predictor/crop_yield_model.pkl')
        scaler = joblib.load('predictor/scaler.pkl')
        poly = joblib.load('predictor/poly.pkl')

        # Preprocess input
        input_data = np.array([[rainfall, temperature, fertilizer]])
        input_scaled = scaler.transform(input_data)
        input_poly = poly.transform(input_scaled)

        # Make prediction
        predicted_yield = model.predict(input_poly)[0]

        # Return predicted yield & model accuracy
        return render(request, 'result.html', {'yield': predicted_yield, 'r2_score': round(r2, 4)})
    
    return render(request, 'index.html')
