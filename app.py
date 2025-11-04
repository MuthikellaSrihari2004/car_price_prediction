# Flask Web Application for Car Price Prediction
# app.py

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and preprocessing objects
print("Loading model and preprocessing objects...")
try:
    model = joblib.load('car_price_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    scaler = joblib.load('scaler.pkl')
    model_metadata = joblib.load('model_metadata.pkl')
    print("✓ Model loaded successfully!")
    print(f"✓ Model: {model_metadata['model_name']}")
    print(f"✓ R² Score: {model_metadata['r2_score']:.4f}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    label_encoders = {}
    scaler = None
    model_metadata = {"model_name": "N/A", "r2_score": 0.0}

# Define valid options for dropdown menus
BRANDS = ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes', 'Audi', 'Hyundai', 'Kia', 'Nissan', 'Chevrolet']
FUEL_TYPES = ['Petrol', 'Diesel', 'Electric', 'Hybrid']
TRANSMISSIONS = ['Manual', 'Automatic']

@app.route('/')
def home():
    """Home page route - displays the prediction form"""
    return render_template('index.html',
                           brands=BRANDS,
                           fuel_types=FUEL_TYPES,
                           transmissions=TRANSMISSIONS,
                           model_info=model_metadata if model else None)

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction route - processes form data and returns price prediction"""
    try:
        # Get form data
        brand = request.form.get('brand')
        model_year = int(request.form.get('model_year'))
        fuel_type = request.form.get('fuel_type')
        transmission = request.form.get('transmission')
        mileage = float(request.form.get('mileage'))
        engine_size = float(request.form.get('engine_size'))

        # Validate inputs
        if not all([brand, model_year, fuel_type, transmission, mileage, engine_size]):
            return render_template('index.html',
                                   brands=BRANDS,
                                   fuel_types=FUEL_TYPES,
                                   transmissions=TRANSMISSIONS,
                                   error="Please fill in all fields")

        if not model:
            return render_template('index.html',
                                   brands=BRANDS,
                                   fuel_types=FUEL_TYPES,
                                   transmissions=TRANSMISSIONS,
                                   error="Model not loaded. Please check your model files.")

        # Create input dataframe
        input_data = pd.DataFrame({
            'Brand': [brand],
            'Model_Year': [model_year],
            'Fuel_Type': [fuel_type],
            'Transmission': [transmission],
            'Mileage': [mileage],
            'Engine_Size': [engine_size]
        })

        # Encode categorical features
        for col in ['Brand', 'Fuel_Type', 'Transmission']:
            if col in label_encoders:
                input_data[col] = label_encoders[col].transform(input_data[col])

        # Scale or predict based on model type
        if model_metadata['model_name'] == 'Linear Regression' and scaler is not None:
            input_data_scaled = scaler.transform(input_data)
            prediction = model.predict(input_data_scaled)[0]
        else:
            prediction = model.predict(input_data)[0]

        # Format prediction
        predicted_price = f"${prediction:,.2f}"
        lower_bound = f"${prediction * 0.9:,.2f}"
        upper_bound = f"${prediction * 1.1:,.2f}"

        return render_template('index.html',
                               brands=BRANDS,
                               fuel_types=FUEL_TYPES,
                               transmissions=TRANSMISSIONS,
                               prediction=predicted_price,
                               lower_bound=lower_bound,
                               upper_bound=upper_bound,
                               input_data={
                                   'brand': brand,
                                   'model_year': model_year,
                                   'fuel_type': fuel_type,
                                   'transmission': transmission,
                                   'mileage': mileage,
                                   'engine_size': engine_size
                               },
                               model_info=model_metadata)

    except Exception as e:
        return render_template('index.html',
                               brands=BRANDS,
                               fuel_types=FUEL_TYPES,
                               transmissions=TRANSMISSIONS,
                               error=f"Prediction error: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access"""
    try:
        data = request.get_json()
        input_data = pd.DataFrame({
            'Brand': [data['brand']],
            'Model_Year': [int(data['model_year'])],
            'Fuel_Type': [data['fuel_type']],
            'Transmission': [data['transmission']],
            'Mileage': [float(data['mileage'])],
            'Engine_Size': [float(data['engine_size'])]
        })

        for col in ['Brand', 'Fuel_Type', 'Transmission']:
            if col in label_encoders:
                input_data[col] = label_encoders[col].transform(input_data[col])

        if model_metadata['model_name'] == 'Linear Regression' and scaler is not None:
            input_data_scaled = scaler.transform(input_data)
            prediction = model.predict(input_data_scaled)[0]
        else:
            prediction = model.predict(input_data)[0]

        return jsonify({
            'success': True,
            'predicted_price': float(prediction),
            'model': model_metadata['model_name'],
            'confidence': {
                'lower': float(prediction * 0.9),
                'upper': float(prediction * 1.1)
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html', model_info=model_metadata)

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚗 CAR PRICE PREDICTION WEB APP")
    print("=" * 60)
    if model:
        print(f"Model: {model_metadata['model_name']}")
        print(f"Performance: R² = {model_metadata['r2_score']:.4f}")
    else:
        print("⚠️ Model not loaded properly — check your .pkl files.")
    print("=" * 60)
    print("\n🌐 Starting Flask server...")
    print("📍 Open your browser and go to: http://127.0.0.1:5000")
    print("\nPress Ctrl+C to stop the server\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
