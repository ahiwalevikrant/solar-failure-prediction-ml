"""
FastAPI application for solar predictive maintenance.
"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

from utils import (
    calculate_panel_temperature,
    calculate_power_output,
    calculate_health_score,
    assess_failure_risk
)

app = FastAPI(
    title="Solar Predictive Maintenance API",
    description="AI-Based Predictive Maintenance for Solar Panels",
    version="1.0.0"
)

# Global model instances
power_model = None
anomaly_detector = None


class PredictionInput(BaseModel):
    """Input schema for power prediction."""
    irradiance: float = Field(..., ge=0, le=1500, description="Solar irradiance in W/m²")
    temperature: float = Field(..., ge=-40, le=60, description="Ambient temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity in %")
    wind_speed: float = Field(..., ge=0, le=50, description="Wind speed in m/s")
    panel_area: Optional[float] = Field(1.6, ge=0.1, le=10, description="Panel area in m²")
    panel_efficiency: Optional[float] = Field(0.18, ge=0.1, le=0.5, description="Panel efficiency")


class PredictionOutput(BaseModel):
    """Output schema for power prediction."""
    predicted_power: float
    panel_temperature: float
    panel_health_score: float
    failure_risk: str
    timestamp: str


class AnomalyInput(BaseModel):
    """Input schema for anomaly detection."""
    irradiance: float
    temperature: float
    humidity: float
    wind_speed: float
    power_output: float


class AnomalyOutput(BaseModel):
    """Output schema for anomaly detection."""
    is_anomaly: bool
    anomaly_score: float
    timestamp: str


class HealthCheckOutput(BaseModel):
    """Output schema for health check."""
    status: str
    model_loaded: bool
    timestamp: str


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global power_model, anomaly_detector
    
    # Try to load models if they exist
    models_dir = "models"
    
    power_model_path = os.path.join(models_dir, "power_model.joblib")
    if os.path.exists(power_model_path):
        try:
            from training import PowerPredictionModel
            power_model = PowerPredictionModel.load(power_model_path)
            print(f"Loaded power model from {power_model_path}")
        except Exception as e:
            print(f"Could not load power model: {e}")
    
    anomaly_model_path = os.path.join(models_dir, "anomaly_model.joblib")
    if os.path.exists(anomaly_model_path):
        try:
            from training import AnomalyDetector
            anomaly_detector = AnomalyDetector.load(anomaly_model_path)
            print(f"Loaded anomaly detector from {anomaly_model_path}")
        except Exception as e:
            print(f"Could not load anomaly detector: {e}")


@app.get("/", response_model=HealthCheckOutput)
async def root():
    """Root endpoint for health check."""
    return HealthCheckOutput(
        status="running",
        model_loaded=power_model is not None,
        timestamp=datetime.now().isoformat()
    )


@app.get("/health", response_model=HealthCheckOutput)
async def health_check():
    """Health check endpoint."""
    return HealthCheckOutput(
        status="healthy" if power_model is not None else "model_not_loaded",
        model_loaded=power_model is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict_power(input_data: PredictionInput):
    """
    Predict solar power output and assess panel health.
    
    Args:
        input_data: Prediction input parameters
        
    Returns:
        Predicted power output and health assessment
    """
    try:
        # Calculate panel temperature
        panel_temp = calculate_panel_temperature(
            ambient_temp=input_data.temperature,
            irradiance=input_data.irradiance,
            wind_speed=input_data.wind_speed
        )
        
        # Calculate predicted power output
        predicted_power = calculate_power_output(
            irradiance=input_data.irradiance,
            panel_temp=panel_temp,
            panel_area=input_data.panel_area,
            efficiency=input_data.panel_efficiency
        )
        
        # Calculate health score based on conditions
        # This is a simplified calculation for demonstration
        temp_factor = 1 - abs(panel_temp - 25) / 100  # Optimal at 25°C
        irradiance_factor = input_data.irradiance / 1000  # Normalize to max irradiance
        humidity_factor = 1 - input_data.humidity / 200  # Lower is better
        
        health_score = calculate_health_score(
            degradation=temp_factor * irradiance_factor,
            anomaly_count=0,
            total_readings=1
        )
        
        # Assess failure risk
        failure_risk = assess_failure_risk(health_score)
        
        return PredictionOutput(
            predicted_power=round(predicted_power, 2),
            panel_temperature=round(panel_temp, 2),
            panel_health_score=round(health_score, 2),
            failure_risk=failure_risk,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect_anomaly", response_model=AnomalyOutput)
async def detect_anomaly(input_data: AnomalyInput):
    """
    Detect anomalies in solar panel performance.
    
    Args:
        input_data: Sensor data for anomaly detection
        
    Returns:
        Anomaly detection results
    """
    try:
        if anomaly_detector is None:
            # Fallback to statistical anomaly detection
            # Using simple z-score based detection
            
            # Expected power calculation (simplified)
            expected_power = (
                input_data.irradiance * 0.18 * 1.6 * 
                (1 - 0.004 * (input_data.temperature - 25))
            )
            
            # Calculate deviation
            deviation = abs(input_data.power_output - expected_power)
            max_deviation = expected_power * 0.3  # 30% threshold
            
            is_anomaly = deviation > max_deviation
            anomaly_score = deviation / expected_power if expected_power > 0 else 0
            
        else:
            # Use trained model
            features = pd.DataFrame([{
                'irradiance': input_data.irradiance,
                'temperature': input_data.temperature,
                'humidity': input_data.humidity,
                'wind_speed': input_data.wind_speed,
                'power_output': input_data.power_output
            }])
            
            predictions, scores = anomaly_detector.predict(features)
            is_anomaly = predictions[0] == -1
            anomaly_score = -scores[0]  # Convert to positive (higher = more anomalous)
        
        return AnomalyOutput(
            is_anomaly=bool(is_anomaly),
            anomaly_score=round(float(anomaly_score), 4),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_info")
async def get_model_info():
    """
    Get information about the loaded models.
    
    Returns:
        Model information
    """
    info = {
        "power_model_loaded": power_model is not None,
        "anomaly_detector_loaded": anomaly_detector is not None,
        "api_version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }
    
    if power_model:
        info["power_model_type"] = power_model.model_type
        info["power_model_features"] = power_model.feature_names
    
    return info


@app.post("/batch_predict")
async def batch_predict(inputs: List[PredictionInput]):
    """
    Make batch predictions for multiple input samples.
    
    Args:
        inputs: List of prediction inputs
        
    Returns:
        List of prediction results
    """
    results = []
    
    for input_data in inputs:
        # Calculate panel temperature
        panel_temp = calculate_panel_temperature(
            ambient_temp=input_data.temperature,
            irradiance=input_data.irradiance,
            wind_speed=input_data.wind_speed
        )
        
        # Calculate predicted power
        predicted_power = calculate_power_output(
            irradiance=input_data.irradiance,
            panel_temp=panel_temp,
            panel_area=input_data.panel_area,
            efficiency=input_data.panel_efficiency
        )
        
        # Health score
        temp_factor = 1 - abs(panel_temp - 25) / 100
        irradiance_factor = input_data.irradiance / 1000
        health_score = calculate_health_score(
            degradation=temp_factor * irradiance_factor,
            anomaly_count=0,
            total_readings=1
        )
        
        failure_risk = assess_failure_risk(health_score)
        
        results.append(PredictionOutput(
            predicted_power=round(predicted_power, 2),
            panel_temperature=round(panel_temp, 2),
            panel_health_score=round(health_score, 2),
            failure_risk=failure_risk,
            timestamp=datetime.now().isoformat()
        ))
    
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
