# AI-Based Predictive Maintenance for Solar Panels

## Overview

This project implements an **AI-driven predictive maintenance system for solar panels**.
The system analyzes historical sensor and environmental data to predict power output, detect anomalies, and identify potential panel degradation before failures occur.

The goal is to improve **solar farm reliability, reduce downtime, and enable proactive maintenance** through machine learning models and real-time monitoring.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Sample Data & Train Models

```bash
python training/train.py
```

This will:
- Generate synthetic solar panel data
- Train the power prediction model (Random Forest)
- Train the anomaly detector
- Train the degradation detector
- Save all models to `models/`

### 3. Start the API Server

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Or simply:

```bash
python api/main.py
```

### 4. Use the API

```bash
# Predict power output
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "irradiance": 820,
    "temperature": 35,
    "humidity": 45,
    "wind_speed": 5
  }'

# Detect anomalies
curl -X POST "http://localhost:8000/detect_anomaly" \
  -H "Content-Type: application/json" \
  -d '{
    "irradiance": 820,
    "temperature": 35,
    "humidity": 45,
    "wind_speed": 5,
    "power_output": 300
  }'

# Health check
curl "http://localhost:8000/health"
```

---

## Key Features

* **Solar Power Prediction**

  * Predicts expected solar power output based on weather and sensor data.

* **Panel Degradation Detection**

  * Detects gradual efficiency loss in solar panels over time.

* **Anomaly Detection**

  * Identifies abnormal system behavior or unexpected drops in power generation.

* **Maintenance Alerts**

  * Generates alerts when system health drops below safe thresholds.

* **ML Model API**

  * Provides prediction endpoints using a FastAPI service.

---

## System Architecture

```
Solar Dataset / Sensors
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Machine Learning Models
        ↓
Model Deployment (FastAPI)
        ↓
Dashboard / Monitoring UI
```

---

## Tech Stack

### Backend

* Python
* FastAPI

### Machine Learning

* Pandas
* NumPy
* Scikit-learn
* PyTorch / TensorFlow

### Visualization

* Matplotlib
* Seaborn

### Frontend

* React

### Deployment

* Docker (optional)

---

## Project Structure

```
solar-predictive-maintenance-ai

data/               # raw and processed datasets
notebooks/          # exploratory data analysis
training/           # ML training scripts
models/             # trained models
api/                # FastAPI prediction service
frontend/           # React dashboard
utils/              # helper functions
```

---

## Machine Learning Components

### Power Prediction Model

Predicts expected solar energy output based on environmental conditions.

### Degradation Detection

Detects long-term panel performance decline.

### Anomaly Detection

Identifies unusual behavior in sensor readings or power output.

Possible algorithms:

* Random Forest
* Gradient Boosting
* LSTM / Transformer models

---

## Example API Endpoint

POST `/predict`

Input:

```
{
  "irradiance": 820,
  "temperature": 35,
  "humidity": 45,
  "wind_speed": 5
}
```

Output:

```
{
  "predicted_power": 410,
  "panel_health_score": 0.92,
  "failure_risk": "low"
}
```

---

## Future Improvements

* Real-time streaming using Kafka
* Transformer-based time series models
* Edge deployment for solar farms
* Automated retraining pipelines

---

## Learning Outcomes

Through this project I explored:

* Predictive maintenance systems
* Time-series analysis for energy systems
* Machine learning model deployment
* End-to-end ML pipelines

---

## Author
Vikrant Ahiwale
## License

MIT License
