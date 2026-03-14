"""
Utility functions for the solar predictive maintenance system.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def calculate_solar_irradiance(
    hour: int,
    day_of_year: int,
    latitude: float = 45.0
) -> float:
    """
    Calculate expected solar irradiance based on time and location.
    
    Args:
        hour: Hour of the day (0-23)
        day_of_year: Day of the year (1-365)
        latitude: Latitude in degrees
        
    Returns:
        Estimated irradiance in W/m²
    """
    # Solar declination angle
    declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
    
    # Hour angle
    hour_angle = 15 * (hour - 12)
    
    # Solar elevation angle
    lat_rad = np.radians(latitude)
    dec_rad = np.radians(declination)
    hour_rad = np.radians(hour_angle)
    
    elevation = np.arcsin(
        np.sin(lat_rad) * np.sin(dec_rad) +
        np.cos(lat_rad) * np.cos(dec_rad) * np.cos(hour_rad)
    )
    
    # Irradiance calculation (simplified)
    if np.degrees(elevation) > 0:
        irradiance = 1361 * np.sin(elevation) * 0.7  # Atmospheric factor
        return max(0, irradiance)
    return 0


def calculate_panel_temperature(
    ambient_temp: float,
    irradiance: float,
    wind_speed: float
) -> float:
    """
    Calculate estimated panel temperature.
    
    Args:
        ambient_temp: Ambient temperature in Celsius
        irradiance: Solar irradiance in W/m²
        wind_speed: Wind speed in m/s
        
    Returns:
        Panel temperature in Celsius
    """
    # Panel temperature model
    noct = 45  # Nominal Operating Cell Temperature
    temp_rise = (noct - 20) * (irradiance / 800)
    wind_cooling = 0.5 * wind_speed
    
    return ambient_temp + temp_rise - wind_cooling


def calculate_power_output(
    irradiance: float,
    panel_temp: float,
    panel_area: float = 1.6,
    efficiency: float = 0.18
) -> float:
    """
    Calculate estimated power output.
    
    Args:
        irradiance: Solar irradiance in W/m²
        panel_temp: Panel temperature in Celsius
        panel_area: Panel area in m²
        efficiency: Panel efficiency
        
    Returns:
        Power output in Watts
    """
    # Temperature coefficient (typically -0.4% to -0.5% per °C)
    temp_coeff = -0.004
    temp_factor = 1 + temp_coeff * (panel_temp - 25)
    
    power = irradiance * panel_area * efficiency * temp_factor
    return max(0, power)


def calculate_degradation_score(
    historical_output: np.ndarray,
    expected_output: np.ndarray
) -> float:
    """
    Calculate panel degradation score.
    
    Args:
        historical_output: Actual historical power output
        expected_output: Expected power output under ideal conditions
        
    Returns:
        Degradation score (0-1, where 1 = no degradation)
    """
    if len(historical_output) == 0:
        return 1.0
    
    ratio = historical_output / (expected_output + 1e-6)
    # Remove outliers
    valid_ratio = ratio[(ratio > 0.5) & (ratio < 1.5)]
    
    if len(valid_ratio) == 0:
        return 1.0
    
    return np.median(valid_ratio)


def detect_anomaly(
    value: float,
    mean: float,
    std: float,
    threshold: float = 3.0
) -> bool:
    """
    Detect if a value is anomalous based on z-score.
    
    Args:
        value: Current value
        mean: Historical mean
        std: Historical standard deviation
        threshold: Z-score threshold
        
    Returns:
        True if anomalous, False otherwise
    """
    if std == 0:
        return False
    
    z_score = abs((value - mean) / std)
    return z_score > threshold


def calculate_health_score(
    degradation: float,
    anomaly_count: int,
    total_readings: int
) -> float:
    """
    Calculate overall panel health score.
    
    Args:
        degradation: Degradation score (0-1)
        anomaly_count: Number of anomalies detected
        total_readings: Total number of readings
        
    Returns:
        Health score (0-1, higher is better)
    """
    degradation_weight = 0.7
    anomaly_weight = 0.3
    
    degradation_score = degradation
    
    if total_readings > 0:
        anomaly_rate = anomaly_count / total_readings
        anomaly_score = 1 - min(anomaly_rate * 10, 1)
    else:
        anomaly_score = 1.0
    
    health_score = (
        degradation_weight * degradation_score +
        anomaly_weight * anomaly_score
    )
    
    return max(0, min(1, health_score))


def assess_failure_risk(health_score: float) -> str:
    """
    Assess failure risk based on health score.
    
    Args:
        health_score: Health score (0-1)
        
    Returns:
        Risk level: 'low', 'medium', 'high'
    """
    if health_score >= 0.75:
        return "low"
    elif health_score >= 0.5:
        return "medium"
    else:
        return "high"


def preprocess_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw sensor data.
    
    Args:
        df: Raw sensor data
        
    Returns:
        Preprocessed dataframe
    """
    df = df.copy()
    
    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from sensor data.
    
    Args:
        df: Preprocessed sensor data
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Time-based features
    if 'timestamp' in df.columns:
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_year'] = pd.to_datetime(df['timestamp']).dt.dayofyear
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
    
    # Rolling statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['hour', 'day_of_year', 'month']:
            df[f'{col}_rolling_mean'] = df[col].rolling(window=7, min_periods=1).mean()
            df[f'{col}_rolling_std'] = df[col].rolling(window=7, min_periods=1).std()
    
    return df
