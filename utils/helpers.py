"""
Helper functions for data processing and visualization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np


def plot_power_prediction(
    actual: np.ndarray,
    predicted: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot actual vs predicted power output.
    
    Args:
        actual: Actual power values
        predicted: Predicted power values
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual', alpha=0.7)
    plt.plot(predicted, label='Predicted', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Power Output (W)')
    plt.title('Solar Power Prediction: Actual vs Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(
    features: List[str],
    importance: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot feature importance from ML model.
    
    Args:
        features: List of feature names
        importance: Feature importance values
        save_path: Path to save the plot
    """
    # Sort by importance
    indices = np.argsort(importance)[::-1]
    sorted_features = [features[i] for i in indices]
    sorted_importance = importance[indices]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_features)), sorted_importance)
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_degradation_trend(
    dates: pd.DatetimeIndex,
    efficiency: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot panel degradation over time.
    
    Args:
        dates: Date index
        efficiency: Efficiency values
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, efficiency, marker='o', linestyle='-', markersize=3)
    plt.xlabel('Date')
    plt.ylabel('Efficiency (%)')
    plt.title('Panel Degradation Over Time')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_anomalies(
    timestamps: pd.DatetimeIndex,
    values: np.ndarray,
    anomalies: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot time series with anomaly highlights.
    
    Args:
        timestamps: Time index
        values: Sensor values
        anomalies: Boolean array indicating anomalies
        save_path: Path to save the plot
    """
    plt.figure(figsize=(14, 6))
    plt.plot(timestamps, values, label='Normal', alpha=0.7)
    plt.scatter(
        timestamps[anomalies],
        values[anomalies],
        color='red',
        s=50,
        label='Anomaly',
        zorder=5
    )
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Anomaly Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_correlation_matrix(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Plot correlation matrix heatmap.
    
    Args:
        df: DataFrame with numerical features
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_model_comparison(
    models: List[str],
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot comparison of multiple models.
    
    Args:
        models: List of model names
        metrics: Dictionary of metric names and values
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, (metric_name, values) in zip(axes, metrics.items()):
        ax.bar(models, values)
        ax.set_title(metric_name)
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray
) -> dict:
    """
    Calculate regression metrics.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    
    # R² score
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # MAPE
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }
