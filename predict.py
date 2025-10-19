# predict.py
# Command-line interface for making predictions

import numpy as np
import joblib
import argparse

def load_model():
    """Load the trained model and encoders"""
    try:
        model = joblib.load("models/flood_model.pkl")
        encoders = joblib.load("models/label_encoders.pkl")
        metadata = joblib.load("models/model_metadata.pkl")
        
        if metadata.get('uses_scaling', False):
            scaler = joblib.load("models/scaler.pkl")
            return model, encoders, metadata, scaler
        
        return model, encoders, metadata, None
    except FileNotFoundError as e:
        print(f"❌ Error: Model files not found. Please run main.py first to train the model.")
        print(f"   Details: {e}")
        exit(1)

def predict_flood_risk(model, encoders, metadata, scaler, input_data):
    """Make a flood risk prediction"""
    
    # Ensure input data is in correct order
    feature_order = metadata['feature_columns']
    
    # Convert to numpy array
    input_array = np.array(input_data).reshape(1, -1)
    
    # Scale if necessary
    if scaler is not None:
        input_array = scaler.transform(input_array)
    
    # Predict
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0]
    
    return prediction, probability

def main():
    parser = argparse.ArgumentParser(description='Flood Risk Prediction System')
    parser.add_argument('--lat', type=float, required=True, help='Latitude')
    parser.add_argument('--lon', type=float, required=True, help='Longitude')
    parser.add_argument('--rainfall', type=float, required=True, help='Rainfall in mm')
    parser.add_argument('--temp', type=float, required=True, help='Temperature in °C')
    parser.add_argument('--humidity', type=float, required=True, help='Humidity in %')
    parser.add_argument('--discharge', type=float, required=True, help='River Discharge in m³/s')
    parser.add_argument('--water-level', type=float, required=True, help='Water Level in m')
    parser.add_argument('--elevation', type=float, required=True, help='Elevation in m')
    parser.add_argument('--land-cover', type=str, required=True, 
                       choices=['Agriculture', 'Forest', 'Urban', 'Water Bodies', 'Desert'],
                       help='Land Cover type')
    parser.add_argument('--soil-type', type=str, required=True,
                       choices=['Clay', 'Loam', 'Sandy', 'Silt', 'Peat'],
                       help='Soil Type')
    parser.add_argument('--population', type=float, required=True, help='Population')
    parser.add_argument('--infrastructure', type=int, choices=[0, 1], required=True,
                       help='Infrastructure (0=No, 1=Yes)')
    parser.add_argument('--historical', type=int, choices=[0, 1], required=True,
                       help='Historical Flooding (0=No, 1=Yes)')
    
    args = parser.parse_args()
    
    # Load model
    print("🔄 Loading model...")
    model, encoders, metadata, scaler = load_model()
    print("✅ Model loaded successfully!\n")
    
    # Prepare input data
    land_cover_encoded = encoders['Land Cover'].transform([args.land_cover])[0]
    soil_type_encoded = encoders['Soil Type'].transform([args.soil_type])[0]
    
    input_data = [
        args.lat, args.lon, args.rainfall, args.temp, args.humidity,
        args.discharge, args.water_level, args.elevation,
        land_cover_encoded, soil_type_encoded,
        args.population, args.infrastructure, args.historical
    ]
    
    # Make prediction
    print("🔍 Making prediction...")
    prediction, probability = predict_flood_risk(model, encoders, metadata, scaler, input_data)
    
    # Display results
    print("\n" + "="*60)
    print("📊 FLOOD RISK PREDICTION RESULTS")
    print("="*60)
    print(f"\nLocation: ({args.lat}, {args.lon})")
    print(f"Rainfall: {args.rainfall} mm")
    print(f"Water Level: {args.water_level} m")
    print(f"Land Cover: {args.land_cover}")
    print(f"Soil Type: {args.soil_type}")
    print("\n" + "-"*60)
    
    if prediction == 1:
        print(f"⚠️  PREDICTION: HIGH FLOOD RISK")
        print(f"🔴 Flood Probability: {probability[1]*100:.2f}%")
        print(f"🟢 Safe Probability: {probability[0]*100:.2f}%")
        print("\n⚠️  RECOMMENDED ACTIONS:")
        print("   • Alert local authorities immediately")
        print("   • Prepare evacuation plan")
        print("   • Monitor water levels closely")
        print("   • Move to higher ground if necessary")
    else:
        print(f"✅ PREDICTION: LOW FLOOD RISK")
        print(f"🟢 Safe Probability: {probability[0]*100:.2f}%")
        print(f"🔴 Flood Probability: {probability[1]*100:.2f}%")
        print("\n✅ Area appears safe from flooding")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()