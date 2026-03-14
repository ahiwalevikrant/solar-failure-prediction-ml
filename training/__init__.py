"""
ML Training module for solar predictive maintenance.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from typing import Tuple, Dict, Any, Optional
import os


class PowerPredictionModel:
    """Model for predicting solar power output."""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the power prediction model.
        
        Args:
            model_type: Type of model ('random_forest' or 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
    def _create_model(self):
        """Create the underlying model based on type."""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                min_samples_split=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the power prediction model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Fraction of data for testing
            
        Returns:
            Dictionary of training metrics
        """
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=5, scoring='r2'
        )
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        self.is_fitted = True
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.
        
        Returns:
            Array of feature importance values
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        return self.model.feature_importances_
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'PowerPredictionModel':
        """
        Load a model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded PowerPredictionModel instance
        """
        data = joblib.load(path)
        instance = cls(model_type=data['model_type'])
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.feature_names = data['feature_names']
        instance.is_fitted = True
        return instance


class AnomalyDetector:
    """Model for detecting anomalies in solar panel performance."""
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies
        """
        from sklearn.ensemble import IsolationForest
        
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def train(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the anomaly detector.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Training results
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        
        # Get anomaly scores
        anomaly_scores = self.model.score_samples(X_scaled)
        
        self.is_fitted = True
        
        return {
            'mean_score': anomaly_scores.mean(),
            'min_score': anomaly_scores.min(),
            'max_score': anomaly_scores.max(),
            'anomaly_count': np.sum(self.model.predict(X_scaled) == -1)
        }
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in new data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Tuple of (predictions, anomaly_scores)
                predictions: -1 for anomaly, 1 for normal
                anomaly_scores: continuous anomaly scores (lower = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        scores = self.model.score_samples(X_scaled)
        
        return predictions, scores
    
    def save(self, path: str) -> None:
        """Save the model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'AnomalyDetector':
        """Load a model from disk."""
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.is_fitted = True
        return instance


class DegradationDetector:
    """Model for detecting panel degradation over time."""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize the degradation detector.
        
        Args:
            window_size: Rolling window size for analysis
        """
        self.window_size = window_size
        self.baseline_efficiency = None
        
    def fit(self, efficiency_data: pd.Series) -> float:
        """
        Fit the baseline efficiency.
        
        Args:
            efficiency_data: Historical efficiency values
            
        Returns:
            Baseline efficiency
        """
        # Use the first window as baseline
        self.baseline_efficiency = efficiency_data.head(self.window_size).mean()
        return self.baseline_efficiency
    
    def detect(
        self,
        efficiency_data: pd.Series
    ) -> pd.DataFrame:
        """
        Detect degradation from efficiency data.
        
        Args:
            efficiency_data: Time series of efficiency values
            
        Returns:
            DataFrame with degradation analysis
        """
        if self.baseline_efficiency is None:
            raise ValueError("Model must be fitted first")
        
        # Calculate rolling metrics
        rolling_mean = efficiency_data.rolling(
            window=self.window_size, min_periods=1
        ).mean()
        
        rolling_std = efficiency_data.rolling(
            window=self.window_size, min_periods=1
        ).std()
        
        # Calculate degradation
        degradation = 1 - (rolling_mean / self.baseline_efficiency)
        
        # Detect significant drops
        significant_drop = degradation < -0.1  # 10% below baseline
        
        results = pd.DataFrame({
            'efficiency': efficiency_data,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'degradation': degradation,
            'significant_drop': significant_drop
        })
        
        return results
    
    def get_degradation_rate(
        self,
        efficiency_data: pd.Series
    ) -> float:
        """
        Calculate the rate of degradation per year.
        
        Args:
            efficiency_data: Time series of efficiency values
            
        Returns:
            Annual degradation rate as percentage
        """
        if len(efficiency_data) < 2:
            return 0.0
        
        # Linear regression for trend
        x = np.arange(len(efficiency_data))
        coefficients = np.polyfit(x, efficiency_data, 1)
        slope = coefficients[0]
        
        # Estimate yearly degradation
        # Assuming daily data, approximately 365 points per year
        points_per_year = len(efficiency_data) / ((efficiency_data.index[-1] - efficiency_data.index[0]).days / 365)
        yearly_degradation = slope * points_per_year
        
        return (yearly_degradation / efficiency_data.mean()) * 100
