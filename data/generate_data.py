"""
Generate sample solar panel data for training and testing.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


def generate_solar_data(
    n_samples: int = 10000,
    n_panels: int = 10,
    start_date: str = "2023-01-01",
    include_anomalies: bool = True,
    include_degradation: bool = True
) -> pd.DataFrame:
    """
    Generate synthetic solar panel data.
    
    Args:
        n_samples: Number of samples to generate
        n_panels: Number of solar panels
        start_date: Start date for the data
        include_anomalies: Whether to include anomalous readings
        include_degradation: Whether to simulate panel degradation
        
    Returns:
        DataFrame with synthetic solar panel data
    """
    np.random.seed(42)
    
    # Generate timestamps
    start = pd.to_datetime(start_date)
    timestamps = [start + timedelta(hours=i) for i in range(n_samples)]
    
    data = []
    
    for panel_id in range(n_panels):
        # Panel-specific parameters
        base_efficiency = np.random.uniform(0.16, 0.20)
        degradation_rate = np.random.uniform(0.001, 0.005) if include_degradation else 0
        anomaly_probability = 0.02 if include_anomalies else 0
        
        for i, timestamp in enumerate(timestamps):
            # Time-based features
            hour = timestamp.hour
            day_of_year = timestamp.dayofyear
            month = timestamp.month
            
            # Solar irradiance (based on time of day and season)
            if 6 <= hour <= 20:
                # Daytime - calculate irradiance
                hour_factor = np.sin(np.pi * (hour - 6) / 14)
                season_factor = 1 + 0.2 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
                irradiance = 1000 * hour_factor * season_factor * np.random.uniform(0.8, 1.0)
            else:
                irradiance = 0
            
            # Environmental conditions
            temperature = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            temperature += np.random.normal(0, 3)
            temperature = np.clip(temperature, -10, 45)
            
            humidity = 50 + 20 * np.sin(2 * np.pi * (day_of_year - 1) / 365)
            humidity += np.random.normal(0, 10)
            humidity = np.clip(humidity, 10, 100)
            
            wind_speed = np.random.exponential(3)
            wind_speed = np.clip(wind_speed, 0, 15)
            
            # Calculate panel temperature
            noct = 45
            temp_rise = (noct - 20) * (irradiance / 800)
            wind_cooling = 0.5 * wind_speed
            panel_temp = temperature + temp_rise - wind_cooling
            
            # Apply degradation over time
            days_elapsed = i / 24  # Assuming hourly data
            current_efficiency = base_efficiency * (1 - degradation_rate) ** (days_elapsed / 365)
            
            # Calculate expected power output
            temp_coeff = -0.004
            temp_factor = 1 + temp_coeff * (panel_temp - 25)
            expected_power = irradiance * 1.6 * current_efficiency * temp_factor
            expected_power = max(0, expected_power)
            
            # Add some noise
            power_output = expected_power * np.random.uniform(0.95, 1.05)
            
            # Introduce anomalies
            is_anomaly = False
            if np.random.random() < anomaly_probability:
                # Random equipment failure
                power_output *= np.random.uniform(0.3, 0.7)
                is_anomaly = True
            
            # Add sensor noise
            irradiance += np.random.normal(0, 10)
            temperature += np.random.normal(0, 0.5)
            humidity += np.random.normal(0, 2)
            wind_speed += np.random.normal(0, 0.5)
            
            # Ensure non-negative values
            irradiance = max(0, irradiance)
            temperature = max(-40, temperature)
            humidity = max(0, min(100, humidity))
            wind_speed = max(0, wind_speed)
            power_output = max(0, power_output)
            
            data.append({
                'timestamp': timestamp,
                'panel_id': f'PANEL_{panel_id:03d}',
                'irradiance': round(irradiance, 2),
                'temperature': round(temperature, 2),
                'humidity': round(humidity, 2),
                'wind_speed': round(wind_speed, 2),
                'panel_temperature': round(panel_temp, 2),
                'power_output': round(power_output, 2),
                'expected_power': round(expected_power, 2),
                'efficiency': round(power_output / (irradiance * 1.6 + 1), 4),
                'is_anomaly': is_anomaly
            })
    
    df = pd.DataFrame(data)
    return df


def generate_anomaly_data(n_normal: int = 1000, n_anomaly: int = 100) -> pd.DataFrame:
    """
    Generate labeled data for anomaly detection training.
    
    Args:
        n_normal: Number of normal samples
        n_anomaly: Number of anomalous samples
        
    Returns:
        DataFrame with labeled anomaly data
    """
    np.random.seed(42)
    
    # Normal data
    normal_data = []
    for _ in range(n_normal):
        irradiance = np.random.uniform(200, 1000)
        temperature = np.random.uniform(15, 35)
        humidity = np.random.uniform(30, 70)
        wind_speed = np.random.uniform(1, 8)
        
        panel_temp = temperature + (irradiance / 800) * 25 - 0.5 * wind_speed
        expected_power = irradiance * 1.6 * 0.18 * (1 - 0.004 * (panel_temp - 25))
        power_output = expected_power * np.random.uniform(0.92, 1.08)
        
        normal_data.append({
            'irradiance': irradiance,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'power_output': power_output,
            'is_anomaly': 0
        })
    
    # Anomalous data
    anomaly_data = []
    for _ in range(n_anomaly):
        irradiance = np.random.uniform(200, 1000)
        temperature = np.random.uniform(15, 35)
        humidity = np.random.uniform(30, 70)
        wind_speed = np.random.uniform(1, 8)
        
        panel_temp = temperature + (irradiance / 800) * 25 - 0.5 * wind_speed
        expected_power = irradiance * 1.6 * 0.18 * (1 - 0.004 * (panel_temp - 25))
        
        # Various types of anomalies
        anomaly_type = np.random.choice([
            'low_output',    # Significantly lower power than expected
            'high_output',   # Significantly higher power (sensor fault)
            'no_power',      # No power output despite good conditions
            'spike'          # Sudden spike in power
        ])
        
        if anomaly_type == 'low_output':
            power_output = expected_power * np.random.uniform(0.2, 0.5)
        elif anomaly_type == 'high_output':
            power_output = expected_power * np.random.uniform(1.3, 1.6)
        elif anomaly_type == 'no_power':
            power_output = expected_power * np.random.uniform(0, 0.1)
        else:  # spike
            power_output = expected_power * np.random.uniform(1.5, 2.0)
        
        anomaly_data.append({
            'irradiance': irradiance,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'power_output': power_output,
            'is_anomaly': 1
        })
    
    df = pd.DataFrame(normal_data + anomaly_data)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
    
    return df


def save_data(df: pd.DataFrame, filename: str, data_dir: str = "data/raw") -> str:
    """
    Save DataFrame to CSV.
    
    Args:
        df: DataFrame to save
        filename: Output filename
        data_dir: Data directory
        
    Returns:
        Path to saved file
    """
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    df.to_csv(filepath, index=False)
    return filepath


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(filepath)


if __name__ == "__main__":
    # Generate sample data
    print("Generating solar panel data...")
    df = generate_solar_data(n_samples=8760, n_panels=5)  # One year of hourly data for 5 panels
    save_data(df, "solar_panel_data.csv")
    print(f"Generated {len(df)} samples")
    
    print("\nGenerating anomaly detection data...")
    anomaly_df = generate_anomaly_data()
    save_data(anomaly_df, "anomaly_data.csv")
    print(f"Generated {len(anomaly_df)} samples ({anomaly_df['is_anomaly'].sum()} anomalies)")
    
    print("\nData generation complete!")
