# Landslide Prediction AI MVP

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from fastapi import FastAPI
import streamlit as st
import plotly.express as px

# Data Collection
def collect_data():
    # Placeholder for data collection logic
    data = {
        'rainfall': np.random.normal(100, 30, 1000),
        'soil_moisture': np.random.normal(0.3, 0.1, 1000),
        'slope_angle': np.random.normal(30, 10, 1000),
        'landslide_occurred': np.random.choice([0, 1], 1000, p=[0.9, 0.1])
    }
    return pd.DataFrame(data)

# Model Training
def train_model(data):
    X = data[['rainfall', 'soil_moisture', 'slope_angle']]
    y = data['landslide_occurred']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# API Setup
app = FastAPI()

@app.post("/predict")
async def predict_risk(rainfall: float, soil_moisture: float, slope_angle: float):
    # Make prediction using the trained model
    prediction = model.predict([[rainfall, soil_moisture, slope_angle]])
    risk_score = float(model.predict_proba([[rainfall, soil_moisture, slope_angle]])[0][1])
    
    return {
        "risk_score": risk_score,
        "risk_level": "High" if risk_score > 0.7 else "Medium" if risk_score > 0.3 else "Low"
    }

# Dashboard
def create_dashboard():
    st.title("Landslide Risk Prediction Dashboard")
    
    rainfall = st.slider("Rainfall (mm)", 0, 300, 100)
    soil_moisture = st.slider("Soil Moisture (%)", 0.0, 1.0, 0.3)
    slope_angle = st.slider("Slope Angle (degrees)", 0, 90, 30)
    
    if st.button("Predict Risk"):
        risk_data = predict_risk(rainfall, soil_moisture, slope_angle)
        st.write(f"Risk Score: {risk_data['risk_score']:.2f}")
        st.write(f"Risk Level: {risk_data['risk_level']}")
        
        # Visualization
        fig = px.gauge(value=risk_data['risk_score'], 
                      range=[0, 1],
                      title="Landslide Risk Meter")
        st.plotly_chart(fig)

# Initialize the system
if __name__ == "__main__":
    data = collect_data()
    model = train_model(data)
    create_dashboard()
    
    # Initialize git repository if not already initialized
    if not os.path.exists('.git'):
        os.system('git init')
    
    # Add files to git
    os.system('git add .')
    
    # Commit changes
    os.system('git commit -m "Initial commit: Landslide Risk Prediction System"')
    
    # Add remote repository (uncomment and update with your repository URL)
    # os.system('git remote add origin https://github.com/leosiby04/landslide-prediction.git')
    
    # Push to GitHub (uncomment after adding remote)
    # os.system('git push -u origin main')
    
    print("\nLandslide Risk Prediction System")
    print("================================")
    print("An open-source project for predicting landslide risks using machine learning")
    print("Repository: https://github.com/leosiby04/landslide-prediction")
    print("Please uncomment and update the git remote and push commands with your repository URL")
    
    # Generate sample predictions to demonstrate capabilities
    print("\nGenerating sample predictions for high-risk scenarios...")
    
    test_scenarios = [
        {"rainfall": 250, "soil_moisture": 0.9, "slope_angle": 75},
        {"rainfall": 200, "soil_moisture": 0.8, "slope_angle": 60},
        {"rainfall": 180, "soil_moisture": 0.85, "slope_angle": 65}
    ]
    
    print("\nSample Risk Assessment Results:")
    print("--------------------------------")
    for scenario in test_scenarios:
        risk = predict_risk(
            rainfall=scenario["rainfall"],
            soil_moisture=scenario["soil_moisture"], 
            slope_angle=scenario["slope_angle"]
        )
        print(f"\nScenario:")
        print(f"Rainfall: {scenario['rainfall']}mm")
        print(f"Soil Moisture: {scenario['soil_moisture']*100}%") 
        print(f"Slope Angle: {scenario['slope_angle']}°")
        print(f"Risk Score: {risk['risk_score']:.2f}")
        print(f"Risk Level: {risk['risk_level']}")
    
    print("\nModel Performance Metrics:")
    print("-------------------------")
    # Calculate and display model metrics
    X = data[['rainfall', 'soil_moisture', 'slope_angle']]
    y = data['landslide_occurred']
    y_pred = model.predict(X)
    print(f"Accuracy: {accuracy_score(y, y_pred):.2f}")
    print(f"Precision: {precision_score(y, y_pred):.2f}")
    print(f"Recall: {recall_score(y, y_pred):.2f}")
    print(f"F1 Score: {f1_score(y, y_pred):.2f}")
    
    print("\nFeature Importance:")
    print("------------------")
    features = ['Rainfall', 'Soil Moisture', 'Slope Angle']
    importances = model.feature_importances_
    for feature, importance in zip(features, importances):
        print(f"{feature}: {importance:.2f}")
    # Fetch historical rainfall data and make predictions
    print("\nRainfall Pattern Analysis:")
    print("-------------------------")
    
    try:
        # Using OpenWeatherMap API for demonstration
        # In production, replace with your actual API key
        API_KEY = "your_api_key_here"
        LAT = "12.9716" # Example coordinates 
        LON = "77.5946"
        
        # Get historical rainfall data
        def get_historical_rainfall():
            base_url = f"http://api.openweathermap.org/data/2.5/forecast"
            params = {
                "lat": LAT,
                "lon": LON,
                "appid": API_KEY,
                "units": "metric"
            }
            
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                rainfall_pattern = []
                for item in data['list']:
                    if 'rain' in item:
                        rainfall_pattern.append(item['rain'].get('3h', 0))
                    else:
                        rainfall_pattern.append(0)
                return rainfall_pattern
            return None

        rainfall_data = get_historical_rainfall()
        if rainfall_data:
            # Analyze rainfall patterns
            avg_rainfall = sum(rainfall_data) / len(rainfall_data)
            max_rainfall = max(rainfall_data)
            
            print(f"\nHistorical Rainfall Analysis:")
            print(f"Average rainfall: {avg_rainfall:.2f}mm")
            print(f"Maximum rainfall: {max_rainfall:.2f}mm")
            
            # Simple prediction for next 24 hours
            if avg_rainfall > 100:
                print("\nWARNING: High rainfall predicted in next 24 hours")
                print("Increased landslide risk potential")
            elif avg_rainfall > 50:
                print("\nMODERATE: Medium rainfall predicted in next 24 hours")
                print("Monitor conditions closely")
            else:
                print("\nNORMAL: Low rainfall predicted in next 24 hours")
                print("Standard vigilance advised")
                
    except Exception as e:
        print(f"\nError fetching rainfall data: {str(e)}")
        print("Please ensure API key is configured correctly")
