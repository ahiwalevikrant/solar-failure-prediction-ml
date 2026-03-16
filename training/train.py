"""
Train ML models for solar predictive maintenance.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import PowerPredictionModel, AnomalyDetector, DegradationDetector
from utils.helpers import (
    plot_power_prediction,
    plot_feature_importance,
    plot_correlation_matrix,
    calculate_metrics
)


def prepare_power_prediction_data(df: pd.DataFrame) -> tuple:
    """
    Prepare data for power prediction model.
    
    Args:
        df: Raw solar panel data
        
    Returns:
        Tuple of (X, y) features and target
    """
    # Feature columns
    features = ['irradiance', 'temperature', 'humidity', 'wind_speed', 'panel_temperature']
    
    X = df[features].copy()
    y = df['power_output'].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    return X, y


def prepare_anomaly_data(df: pd.DataFrame) -> tuple:
    """
    Prepare data for anomaly detection model.
    
    Args:
        df: Labeled anomaly data
        
    Returns:
        Tuple of (X, y) features and labels
    """
    features = ['irradiance', 'temperature', 'humidity', 'wind_speed', 'power_output']
    
    X = df[features].copy()
    y = df['is_anomaly'].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    return X, y


def train_power_model(
    data_path: str = "data/raw/solar_panel_data.csv",
    model_type: str = "random_forest",
    output_dir: str = "models"
) -> dict:
    """
    Train the power prediction model.
    
    Args:
        data_path: Path to training data
        model_type: Type of model to train
        output_dir: Directory to save models
        
    Returns:
        Training metrics
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print("Preparing features...")
    X, y = prepare_power_prediction_data(df)
    
    print(f"Training {model_type} model...")
    model = PowerPredictionModel(model_type=model_type)
    metrics = model.train(X, y)
    
    # Print metrics
    print("\n=== Training Metrics ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "power_model.joblib")
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Plot feature importance
    importance = model.get_feature_importance()
    plot_feature_importance(
        features=X.columns.tolist(),
        importance=importance,
        save_path=os.path.join(output_dir, "feature_importance.png")
    )
    print(f"Feature importance plot saved")
    
    return metrics


def train_anomaly_detector(
    data_path: str = "data/raw/anomaly_data.csv",
    output_dir: str = "models"
) -> dict:
    """
    Train the anomaly detection model.
    
    Args:
        data_path: Path to training data
        output_dir: Directory to save models
        
    Returns:
        Training metrics
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print("Preparing features...")
    X, y = prepare_anomaly_data(df)
    
    print("Training anomaly detector...")
    detector = AnomalyDetector(contamination=0.1)
    results = detector.train(X)
    
    # Print results
    print("\n=== Training Results ===")
    for key, value in results.items():
        print(f"{key}: {value}")
    
    # Evaluate on labeled data
    predictions, scores = detector.predict(X)
    accuracy = (predictions == y.values).mean()
    print(f"Accuracy on labeled data: {accuracy:.4f}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "anomaly_model.joblib")
    detector.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    return results


def train_degradation_detector(
    data_path: str = "data/raw/solar_panel_data.csv",
    output_dir: str = "models"
) -> dict:
    """
    Train the degradation detection model.
    
    Args:
        data_path: Path to training data
        output_dir: Directory to save models
        
    Returns:
        Training metrics
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Get a single panel's data
    panel_data = df[df['panel_id'] == df['panel_id'].iloc[0]].copy()
    panel_data = panel_data.set_index('timestamp')
    panel_data.index = pd.to_datetime(panel_data.index)
    panel_data = panel_data.sort_index()
    
    print("Training degradation detector...")
    detector = DegradationDetector(window_size=30)
    baseline = detector.fit(panel_data['efficiency'])
    
    print(f"Baseline efficiency: {baseline:.4f}")
    
    # Detect degradation
    results = detector.detect(panel_data['efficiency'])
    
    # Calculate degradation rate
    degradation_rate = detector.get_degradation_rate(panel_data['efficiency'])
    print(f"Annual degradation rate: {degradation_rate:.2f}%")
    
    # Plot degradation
    plt.figure(figsize=(12, 6))
    plt.plot(panel_data.index, results['degradation'] * 100)
    plt.xlabel('Date')
    plt.ylabel('Degradation (%)')
    plt.title('Panel Degradation Over Time')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "degradation_trend.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Degradation plot saved")
    
    return {
        'baseline_efficiency': baseline,
        'annual_degradation_rate': degradation_rate
    }


def main():
    """Main training pipeline."""
    print("=" * 50)
    print("Solar Predictive Maintenance - Model Training")
    print("=" * 50)
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Generate sample data if not exists
    if not os.path.exists("data/raw/solar_panel_data.csv"):
        print("\nGenerating sample data...")
        from data.generate_data import generate_solar_data, generate_anomaly_data, save_data
        
        df = generate_solar_data(n_samples=8760, n_panels=5)
        save_data(df, "solar_panel_data.csv")
        
        anomaly_df = generate_anomaly_data()
        save_data(anomaly_df, "anomaly_data.csv")
    
    # Train power prediction model
    print("\n" + "=" * 50)
    print("Training Power Prediction Model")
    print("=" * 50)
    train_power_model()
    
    # Train anomaly detector
    print("\n" + "=" * 50)
    print("Training Anomaly Detector")
    print("=" * 50)
    train_anomaly_detector()
    
    # Train degradation detector
    print("\n" + "=" * 50)
    print("Training Degradation Detector")
    print("=" * 50)
    train_degradation_detector()
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
