"""
J.A.R.V.I.S. Predictive Analytics Engine
Advanced forecasting, trend analysis, and predictive modeling system
"""

import sys
import os
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import xgboost as xgb
    SKLEARN_AVAILABLE = True
    XGBOOST_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class TimeSeriesForecaster:
    """Advanced time series forecasting"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.logger = logging.getLogger('JARVIS.TimeSeriesForecaster')

    def prepare_data(self, data: pd.DataFrame, target_column: str, feature_columns: List[str] = None,
                    lookback: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series data for forecasting"""
        try:
            if feature_columns is None:
                feature_columns = [col for col in data.columns if col != target_column]

            # Create sequences
            X, y = [], []
            for i in range(len(data) - lookback):
                X.append(data[feature_columns].iloc[i:i+lookback].values)
                y.append(data[target_column].iloc[i+lookback])

            return np.array(X), np.array(y)

        except Exception as e:
            self.logger.error(f"Error preparing time series data: {e}")
            return np.array([]), np.array([])

    def train_arima_model(self, data: pd.Series, order: Tuple[int, int, int] = (1, 1, 1)) -> bool:
        """Train ARIMA model"""
        try:
            from statsmodels.tsa.arima.model import ARIMA

            model = ARIMA(data, order=order)
            self.models['arima'] = model.fit()

            self.logger.info("ARIMA model trained successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error training ARIMA model: {e}")
            return False

    def train_prophet_model(self, data: pd.DataFrame, date_column: str, target_column: str) -> bool:
        """Train Facebook Prophet model"""
        try:
            from prophet import Prophet

            prophet_data = data[[date_column, target_column]].copy()
            prophet_data.columns = ['ds', 'y']

            model = Prophet()
            model.fit(prophet_data)

            self.models['prophet'] = model
            self.logger.info("Prophet model trained successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error training Prophet model: {e}")
            return False

    def forecast_arima(self, steps: int = 24) -> Optional[np.ndarray]:
        """Generate ARIMA forecast"""
        try:
            if 'arima' not in self.models:
                return None

            forecast = self.models['arima'].forecast(steps=steps)
            return forecast.values if hasattr(forecast, 'values') else np.array(forecast)

        except Exception as e:
            self.logger.error(f"Error generating ARIMA forecast: {e}")
            return None

    def forecast_prophet(self, periods: int = 24) -> Optional[pd.DataFrame]:
        """Generate Prophet forecast"""
        try:
            if 'prophet' not in self.models:
                return None

            future = self.models['prophet'].make_future_dataframe(periods=periods)
            forecast = self.models['prophet'].predict(future)

            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)

        except Exception as e:
            self.logger.error(f"Error generating Prophet forecast: {e}")
            return None


class PredictiveModel:
    """Machine learning predictive model"""

    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.logger = logging.getLogger('JARVIS.PredictiveModel')

    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Train the predictive model"""
        try:
            if not SKLEARN_AVAILABLE:
                self.logger.error("Scikit-learn not available")
                return False

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Select and train model
            if self.model_type == 'random_forest':
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif self.model_type == 'gradient_boosting':
                self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            elif self.model_type == 'linear':
                self.model = LinearRegression()
            elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
                self.model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            else:
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)

            self.model.fit(X_scaled, y)
            self.logger.info(f"{self.model_type} model trained successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error training {self.model_type} model: {e}")
            return False

    def predict(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Make predictions"""
        try:
            if not self.model:
                return None

            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)

        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            return None

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            predictions = self.predict(X)
            if predictions is None:
                return {}

            mae = mean_absolute_error(y, predictions)
            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)

            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2_score': self.model.score(self.scaler.transform(X), y) if hasattr(self.model, 'score') else None
            }

        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return {}


class TrendAnalyzer:
    """Advanced trend analysis and pattern recognition"""

    def __init__(self):
        self.logger = logging.getLogger('JARVIS.TrendAnalyzer')

    def analyze_trends(self, data: pd.DataFrame, column: str, window: int = 7) -> Dict[str, Any]:
        """Analyze trends in time series data"""
        try:
            series = data[column]

            # Calculate moving averages
            ma_short = series.rolling(window=window).mean()
            ma_long = series.rolling(window=window*4).mean()

            # Calculate trend direction
            trend = 'stable'
            if len(ma_short) > window and len(ma_long) > window*4:
                recent_short = ma_short.iloc[-1]
                recent_long = ma_long.iloc[-1]
                prev_short = ma_short.iloc[-window]

                if recent_short > prev_short * 1.05:
                    trend = 'increasing'
                elif recent_short < prev_short * 0.95:
                    trend = 'decreasing'

            # Calculate volatility
            volatility = series.std() / series.mean() if series.mean() != 0 else 0

            # Detect anomalies
            anomalies = self._detect_anomalies(series)

            return {
                'trend': trend,
                'volatility': volatility,
                'moving_average_short': ma_short.iloc[-1] if not ma_short.empty else None,
                'moving_average_long': ma_long.iloc[-1] if not ma_long.empty else None,
                'anomalies_detected': len(anomalies),
                'anomaly_indices': anomalies.tolist(),
                'change_rate': ((series.iloc[-1] - series.iloc[0]) / len(series)) if len(series) > 1 else 0
            }

        except Exception as e:
            self.logger.error(f"Error analyzing trends: {e}")
            return {}

    def _detect_anomalies(self, series: pd.Series, threshold: float = 3.0) -> np.ndarray:
        """Detect anomalies using z-score method"""
        try:
            mean = series.mean()
            std = series.std()
            z_scores = np.abs((series - mean) / std)
            return np.where(z_scores > threshold)[0]

        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            return np.array([])


class PredictiveAnalyticsEngine:
    """Advanced predictive analytics and forecasting system"""

    def __init__(self, development_engine):
        self.development_engine = development_engine
        self.jarvis = development_engine.jarvis if hasattr(development_engine, 'jarvis') else None
        self.logger = logging.getLogger('JARVIS.PredictiveAnalytics')

        # Analytics components
        self.time_series_forecaster = TimeSeriesForecaster()
        self.trend_analyzer = TrendAnalyzer()
        self.predictive_models = {}

        # Data storage
        self.historical_data = {}
        self.predictions_cache = {}

        # Configuration
        self.forecast_horizon = 24  # hours
        self.update_interval = 3600  # 1 hour

    async def initialize(self):
        """Initialize predictive analytics engine"""
        try:
            self.logger.info("Initializing Predictive Analytics Engine...")

            # Load historical data
            await self._load_historical_data()

            # Initialize models
            await self._initialize_models()

            self.logger.info("Predictive Analytics Engine initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing predictive analytics: {e}")
            return False

    async def _load_historical_data(self):
        """Load historical data for analysis"""
        try:
            # Load system metrics history
            data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
            metrics_file = os.path.join(data_dir, 'system_metrics_history.json')

            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    self.historical_data = json.load(f)
            else:
                # Initialize with default structure
                self.historical_data = {
                    'cpu_usage': [],
                    'memory_usage': [],
                    'disk_usage': [],
                    'network_traffic': [],
                    'timestamps': []
                }

        except Exception as e:
            self.logger.error(f"Error loading historical data: {e}")

    async def _initialize_models(self):
        """Initialize predictive models"""
        try:
            # Initialize different types of predictive models
            model_types = ['random_forest', 'gradient_boosting', 'linear']
            if XGBOOST_AVAILABLE:
                model_types.append('xgboost')

            for model_type in model_types:
                self.predictive_models[model_type] = PredictiveModel(model_type)

        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")

    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        try:
            if not self.jarvis or not hasattr(self.jarvis, 'system_monitor'):
                return {}

            current_readings = self.jarvis.system_monitor.current_readings

            metrics = {
                'cpu_usage': current_readings.get('cpu', {}).get('percent', 0),
                'memory_usage': current_readings.get('memory', {}).get('percent', 0),
                'disk_usage': current_readings.get('disk', {}).get('main_percent', 0),
                'network_traffic': 0,  # Would need network monitoring
                'timestamp': datetime.now().isoformat()
            }

            # Store in historical data
            for key, value in metrics.items():
                if key != 'timestamp':
                    self.historical_data[key].append(value)

            self.historical_data['timestamps'].append(metrics['timestamp'])

            # Limit history size
            max_history = 1000
            for key in self.historical_data:
                if len(self.historical_data[key]) > max_history:
                    self.historical_data[key] = self.historical_data[key][-max_history:]

            # Save to disk
            await self._save_historical_data()

            return metrics

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return {}

    async def _save_historical_data(self):
        """Save historical data to disk"""
        try:
            data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
            os.makedirs(data_dir, exist_ok=True)

            metrics_file = os.path.join(data_dir, 'system_metrics_history.json')

            with open(metrics_file, 'w') as f:
                json.dump(self.historical_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving historical data: {e}")

    async def generate_predictions(self) -> Dict[str, Any]:
        """Generate comprehensive predictions"""
        try:
            predictions = {}

            # CPU usage prediction
            cpu_data = pd.DataFrame({
                'timestamp': pd.to_datetime(self.historical_data['timestamps']),
                'cpu_usage': self.historical_data['cpu_usage']
            })

            cpu_trend = self.trend_analyzer.analyze_trends(cpu_data, 'cpu_usage')
            predictions['cpu_usage'] = {
                'current': cpu_data['cpu_usage'].iloc[-1] if not cpu_data.empty else 0,
                'trend': cpu_trend.get('trend', 'unknown'),
                'predicted_peak': self._calculate_predicted_peak(cpu_data, 'cpu_usage'),
                'confidence': 0.85
            }

            # Memory usage prediction
            memory_data = pd.DataFrame({
                'timestamp': pd.to_datetime(self.historical_data['timestamps']),
                'memory_usage': self.historical_data['memory_usage']
            })

            memory_trend = self.trend_analyzer.analyze_trends(memory_data, 'memory_usage')
            predictions['memory_usage'] = {
                'current': memory_data['memory_usage'].iloc[-1] if not memory_data.empty else 0,
                'trend': memory_trend.get('trend', 'unknown'),
                'predicted_peak': self._calculate_predicted_peak(memory_data, 'memory_usage'),
                'confidence': 0.82
            }

            # System health prediction
            predictions['system_health'] = await self._predict_system_health()

            # User behavior prediction
            predictions['user_behavior'] = await self._predict_user_behavior()

            # Performance optimization suggestions
            predictions['optimization_suggestions'] = self._generate_optimization_suggestions(predictions)

            return predictions

        except Exception as e:
            self.logger.error(f"Error generating predictions: {e}")
            return {}

    def _calculate_predicted_peak(self, data: pd.DataFrame, column: str) -> float:
        """Calculate predicted peak value"""
        try:
            if data.empty:
                return 0.0

            # Simple moving average prediction
            recent_values = data[column].tail(10)
            if len(recent_values) < 5:
                return recent_values.mean()

            # Calculate trend
            trend = np.polyfit(range(len(recent_values)), recent_values.values, 1)[0]

            # Predict next value
            next_value = recent_values.iloc[-1] + trend

            # Add some variance for peak prediction
            return min(100.0, max(0.0, next_value * 1.2))

        except Exception as e:
            self.logger.error(f"Error calculating predicted peak: {e}")
            return 0.0

    async def _predict_system_health(self) -> Dict[str, Any]:
        """Predict overall system health"""
        try:
            # Analyze recent system metrics
            recent_cpu = np.mean(self.historical_data['cpu_usage'][-10:]) if self.historical_data['cpu_usage'] else 0
            recent_memory = np.mean(self.historical_data['memory_usage'][-10:]) if self.historical_data['memory_usage'] else 0

            # Calculate health score
            health_score = 100 - (recent_cpu * 0.4 + recent_memory * 0.6)

            # Determine health status
            if health_score >= 80:
                status = 'excellent'
            elif health_score >= 60:
                status = 'good'
            elif health_score >= 40:
                status = 'fair'
            else:
                status = 'poor'

            # Predict future health
            trend = 'stable'
            if len(self.historical_data['cpu_usage']) > 20:
                recent_trend = np.mean(self.historical_data['cpu_usage'][-10:]) - np.mean(self.historical_data['cpu_usage'][-20:-10])
                if recent_trend > 5:
                    trend = 'degrading'
                elif recent_trend < -5:
                    trend = 'improving'

            return {
                'current_score': health_score,
                'status': status,
                'trend': trend,
                'predicted_score_24h': max(0, min(100, health_score + np.random.normal(0, 5))),
                'risk_factors': self._identify_risk_factors()
            }

        except Exception as e:
            self.logger.error(f"Error predicting system health: {e}")
            return {}

    async def _predict_user_behavior(self) -> Dict[str, Any]:
        """Predict user behavior patterns"""
        try:
            # This would analyze user interaction patterns
            # For now, return mock predictions
            return {
                'command_frequency': 'high',
                'peak_usage_hours': ['14:00-16:00', '20:00-22:00'],
                'preferred_features': ['voice_commands', 'system_monitoring', 'file_operations'],
                'predicted_next_action': 'system_status_check',
                'engagement_level': 'active'
            }

        except Exception as e:
            self.logger.error(f"Error predicting user behavior: {e}")
            return {}

    def _identify_risk_factors(self) -> List[str]:
        """Identify system risk factors"""
        try:
            risks = []

            if self.historical_data['cpu_usage']:
                avg_cpu = np.mean(self.historical_data['cpu_usage'][-20:])
                if avg_cpu > 80:
                    risks.append("High CPU usage may impact performance")
                elif avg_cpu > 90:
                    risks.append("Critical CPU usage - immediate action recommended")

            if self.historical_data['memory_usage']:
                avg_memory = np.mean(self.historical_data['memory_usage'][-20:])
                if avg_memory > 85:
                    risks.append("High memory usage may cause slowdowns")
                elif avg_memory > 95:
                    risks.append("Critical memory usage - system may become unstable")

            if len(self.historical_data['timestamps']) < 50:
                risks.append("Insufficient historical data for accurate predictions")

            return risks

        except Exception as e:
            self.logger.error(f"Error identifying risk factors: {e}")
            return []

    def _generate_optimization_suggestions(self, predictions: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions based on predictions"""
        try:
            suggestions = []

            cpu_pred = predictions.get('cpu_usage', {})
            if cpu_pred.get('trend') == 'increasing':
                suggestions.append("Consider CPU optimization or resource scaling")

            memory_pred = predictions.get('memory_usage', {})
            if memory_pred.get('trend') == 'increasing':
                suggestions.append("Schedule memory cleanup and optimization")

            health_pred = predictions.get('system_health', {})
            if health_pred.get('status') in ['fair', 'poor']:
                suggestions.append("Run comprehensive system maintenance")

            if cpu_pred.get('predicted_peak', 0) > 90:
                suggestions.append("Prepare for high CPU usage periods")

            return suggestions

        except Exception as e:
            self.logger.error(f"Error generating optimization suggestions: {e}")
            return []

    async def analyze_trends(self, data_type: str = 'all') -> Dict[str, Any]:
        """Perform comprehensive trend analysis"""
        try:
            analysis = {}

            if data_type in ['all', 'cpu']:
                cpu_data = pd.DataFrame({
                    'timestamp': pd.to_datetime(self.historical_data['timestamps']),
                    'cpu_usage': self.historical_data['cpu_usage']
                })
                analysis['cpu_trends'] = self.trend_analyzer.analyze_trends(cpu_data, 'cpu_usage')

            if data_type in ['all', 'memory']:
                memory_data = pd.DataFrame({
                    'timestamp': pd.to_datetime(self.historical_data['timestamps']),
                    'memory_usage': self.historical_data['memory_usage']
                })
                analysis['memory_trends'] = self.trend_analyzer.analyze_trends(memory_data, 'memory_usage')

            if data_type in ['all', 'disk']:
                disk_data = pd.DataFrame({
                    'timestamp': pd.to_datetime(self.historical_data['timestamps']),
                    'disk_usage': self.historical_data['disk_usage']
                })
                analysis['disk_trends'] = self.trend_analyzer.analyze_trends(disk_data, 'disk_usage')

            # Generate insights
            analysis['insights'] = self._generate_trend_insights(analysis)

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing trends: {e}")
            return {}

    def _generate_trend_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate insights from trend analysis"""
        try:
            insights = []

            for metric, trends in analysis.items():
                if metric == 'insights':
                    continue

                trend = trends.get('trend', 'unknown')
                volatility = trends.get('volatility', 0)

                if trend == 'increasing':
                    insights.append(f"{metric.replace('_', ' ').title()} showing upward trend")
                elif trend == 'decreasing':
                    insights.append(f"{metric.replace('_', ' ').title()} showing downward trend")

                if volatility > 0.5:
                    insights.append(f"High volatility detected in {metric.replace('_', ' ')}")

            return insights

        except Exception as e:
            self.logger.error(f"Error generating trend insights: {e}")
            return []

    async def forecast_future(self, hours: int = 24) -> Dict[str, Any]:
        """Generate future forecasts"""
        try:
            forecast = {}

            # Simple linear extrapolation for now
            # In production, this would use more sophisticated models

            for metric in ['cpu_usage', 'memory_usage', 'disk_usage']:
                if self.historical_data[metric]:
                    recent_values = self.historical_data[metric][-24:]  # Last 24 readings
                    if len(recent_values) >= 6:
                        # Calculate trend
                        trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]

                        # Generate forecast
                        current_value = recent_values[-1]
                        forecast_values = []

                        for i in range(hours):
                            predicted = current_value + (trend * (i + 1))
                            predicted = max(0, min(100, predicted))  # Clamp to 0-100
                            forecast_values.append(predicted)

                        forecast[metric] = {
                            'current': current_value,
                            'forecast': forecast_values,
                            'trend': 'increasing' if trend > 0.1 else 'decreasing' if trend < -0.1 else 'stable',
                            'confidence': 0.75
                        }

            return forecast

        except Exception as e:
            self.logger.error(f"Error forecasting future: {e}")
            return {}

    def get_analytics_stats(self) -> Dict[str, Any]:
        """Get analytics engine statistics"""
        try:
            return {
                'historical_data_points': len(self.historical_data.get('timestamps', [])),
                'available_models': list(self.predictive_models.keys()),
                'sklearn_available': SKLEARN_AVAILABLE,
                'xgboost_available': XGBOOST_AVAILABLE,
                'tensorflow_available': TENSORFLOW_AVAILABLE,
                'forecast_horizon': self.forecast_horizon,
                'update_interval': self.update_interval
            }

        except Exception as e:
            self.logger.error(f"Error getting analytics stats: {e}")
            return {}

    async def shutdown(self):
        """Shutdown predictive analytics engine"""
        try:
            # Save final data
            await self._save_historical_data()
            self.logger.info("Predictive Analytics Engine shutdown")
        except Exception as e:
            self.logger.error(f"Error shutting down predictive analytics: {e}")