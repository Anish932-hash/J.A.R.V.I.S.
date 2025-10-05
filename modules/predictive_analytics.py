"""
J.A.R.V.I.S. Predictive Analytics
Machine learning-based prediction system for system behavior and user needs
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

# Import ML libraries
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


class PredictiveAnalytics:
    """
    Advanced predictive analytics system
    Uses machine learning to predict system behavior and user needs
    """

    def __init__(self, jarvis_instance):
        """
        Initialize predictive analytics

        Args:
            jarvis_instance: Reference to main JARVIS instance
        """
        self.jarvis = jarvis_instance
        self.logger = logging.getLogger('JARVIS.PredictiveAnalytics')

        # ML models
        self.models = {}
        self.scalers = {}

        # Prediction data
        self.training_data = {
            "cpu_usage": [],
            "memory_usage": [],
            "command_frequency": [],
            "user_activity": [],
            "system_events": []
        }

        # Predictions
        self.predictions = {}

        # Configuration
        self.config = {
            "prediction_horizon": 24,  # hours
            "training_interval": 3600,  # 1 hour
            "min_training_samples": 100,
            "model_update_frequency": 86400,  # 24 hours
            "enable_ml_predictions": SKLEARN_AVAILABLE or TENSORFLOW_AVAILABLE or PYTORCH_AVAILABLE
        }

        # Performance tracking
        self.stats = {
            "predictions_made": 0,
            "predictions_accuracy": 0.0,
            "models_trained": 0,
            "training_cycles": 0
        }

    async def initialize(self):
        """Initialize predictive analytics"""
        try:
            self.logger.info("Initializing predictive analytics...")

            # Load existing training data
            await self._load_training_data()

            # Initialize ML models if available
            if self.config["enable_ml_predictions"]:
                await self._initialize_ml_models()

            # Start prediction updates
            asyncio.create_task(self._prediction_update_loop())

            self.logger.info("Predictive analytics initialized")

        except Exception as e:
            self.logger.error(f"Error initializing predictive analytics: {e}")
            raise

    async def _load_training_data(self):
        """Load existing training data"""
        try:
            data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_data.json')

            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    import json
                    data = json.load(f)

                self.training_data = data.get("training_data", self.training_data)

                self.logger.info(f"Loaded training data: {sum(len(v) for v in self.training_data.values())} samples")

        except Exception as e:
            self.logger.error(f"Error loading training data: {e}")

    async def _initialize_ml_models(self):
        """Initialize machine learning models"""
        try:
            if SKLEARN_AVAILABLE:
                # Initialize scikit-learn models
                self.models["cpu_predictor"] = RandomForestRegressor(n_estimators=100)
                self.models["memory_predictor"] = LinearRegression()
                self.models["command_predictor"] = RandomForestRegressor(n_estimators=50)

                self.scalers["cpu_scaler"] = StandardScaler()
                self.scalers["memory_scaler"] = StandardScaler()
                self.scalers["command_scaler"] = StandardScaler()

                self.logger.info("Initialized scikit-learn models")

            if TENSORFLOW_AVAILABLE:
                # Initialize TensorFlow models
                self.models["tf_cpu_model"] = self._create_tensorflow_model()
                self.logger.info("Initialized TensorFlow models")

            if PYTORCH_AVAILABLE:
                # Initialize PyTorch models
                self.models["torch_cpu_model"] = self._create_pytorch_model()
                self.logger.info("Initialized PyTorch models")

        except Exception as e:
            self.logger.error(f"Error initializing ML models: {e}")

    def _create_tensorflow_model(self):
        """Create TensorFlow prediction model"""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse')
            return model

        except Exception as e:
            self.logger.error(f"Error creating TensorFlow model: {e}")
            return None

    def _create_pytorch_model(self):
        """Create PyTorch prediction model"""
        try:
            class SimplePredictor(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(10, 64)
                    self.fc2 = nn.Linear(64, 32)
                    self.fc3 = nn.Linear(32, 1)

                def forward(self, x):
                    x = torch.relu(self.fc1(x))
                    x = torch.relu(self.fc2(x))
                    x = self.fc3(x)
                    return x

            return SimplePredictor()

        except Exception as e:
            self.logger.error(f"Error creating PyTorch model: {e}")
            return None

    async def collect_training_data(self):
        """Collect training data from system monitoring"""
        try:
            current_time = time.time()

            # Collect current system metrics
            if hasattr(self.jarvis, 'system_monitor'):
                cpu_info = self.jarvis.system_monitor.current_readings.get('cpu', {})
                memory_info = self.jarvis.system_monitor.current_readings.get('memory', {})

                # Add to training data
                self.training_data["cpu_usage"].append({
                    "timestamp": current_time,
                    "value": cpu_info.get('percent', 0),
                    "features": self._extract_features(current_time, "cpu")
                })

                self.training_data["memory_usage"].append({
                    "timestamp": current_time,
                    "value": memory_info.get('percent', 0),
                    "features": self._extract_features(current_time, "memory")
                })

            # Collect command frequency data
            if hasattr(self.jarvis, 'command_processor'):
                command_count = len(self.jarvis.command_processor.command_history)
                self.training_data["command_frequency"].append({
                    "timestamp": current_time,
                    "value": command_count,
                    "features": self._extract_features(current_time, "commands")
                })

            # Maintain data limits
            for key in self.training_data.keys():
                if len(self.training_data[key]) > 10000:  # Keep last 10k samples
                    self.training_data[key] = self.training_data[key][-10000:]

        except Exception as e:
            self.logger.error(f"Error collecting training data: {e}")

    def _extract_features(self, timestamp: float, data_type: str) -> List[float]:
        """Extract features for ML model"""
        try:
            features = []

            # Time-based features
            hour = time.localtime(timestamp).tm_hour
            day_of_week = time.localtime(timestamp).tm_wday
            month = time.localtime(timestamp).tm_mon

            features.extend([hour, day_of_week, month])

            # Historical data features (simplified)
            if data_type == "cpu" and self.training_data["cpu_usage"]:
                recent_cpu = [d["value"] for d in self.training_data["cpu_usage"][-10:]]
                features.extend([
                    np.mean(recent_cpu),
                    np.std(recent_cpu),
                    np.min(recent_cpu),
                    np.max(recent_cpu)
                ])
            elif data_type == "memory" and self.training_data["memory_usage"]:
                recent_memory = [d["value"] for d in self.training_data["memory_usage"][-10:]]
                features.extend([
                    np.mean(recent_memory),
                    np.std(recent_memory),
                    np.min(recent_memory),
                    np.max(recent_memory)
                ])
            else:
                # Default features
                features.extend([0, 0, 0, 0])

            # Pad or truncate to fixed length
            if len(features) < 10:
                features.extend([0] * (10 - len(features)))
            else:
                features = features[:10]

            return features

        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return [0] * 10

    async def train_models(self) -> Dict[str, Any]:
        """Train prediction models"""
        try:
            if not self.config["enable_ml_predictions"]:
                return {"success": False, "error": "ML libraries not available"}

            training_results = {}

            # Train CPU usage model
            if len(self.training_data["cpu_usage"]) >= self.config["min_training_samples"]:
                cpu_result = await self._train_cpu_model()
                training_results["cpu_model"] = cpu_result

            # Train memory usage model
            if len(self.training_data["memory_usage"]) >= self.config["min_training_samples"]:
                memory_result = await self._train_memory_model()
                training_results["memory_model"] = memory_result

            # Train command frequency model
            if len(self.training_data["command_frequency"]) >= self.config["min_training_samples"]:
                command_result = await self._train_command_model()
                training_results["command_model"] = command_result

            self.stats["models_trained"] += len(training_results)
            self.stats["training_cycles"] += 1

            return {
                "success": True,
                "models_trained": len(training_results),
                "training_results": training_results,
                "total_samples": sum(len(data) for data in self.training_data.values())
            }

        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            return {"success": False, "error": str(e)}

    async def _train_cpu_model(self) -> Dict[str, Any]:
        """Train CPU usage prediction model"""
        try:
            if not SKLEARN_AVAILABLE:
                return {"success": False, "error": "scikit-learn not available"}

            # Prepare training data
            data = self.training_data["cpu_usage"]
            if len(data) < 50:  # Need minimum samples
                return {"success": False, "error": "Insufficient training data"}

            # Extract features and targets
            X = [d["features"] for d in data[:-1]]  # All but last
            y = [d["value"] for d in data[1:]]     # Shifted by 1 (predict next)

            # Scale features
            X_scaled = self.scalers["cpu_scaler"].fit_transform(X)

            # Train model
            self.models["cpu_predictor"].fit(X_scaled, y)

            # Calculate training accuracy
            y_pred = self.models["cpu_predictor"].predict(X_scaled)
            mse = mean_squared_error(y, y_pred)

            return {
                "success": True,
                "samples_used": len(X),
                "mse": mse,
                "rmse": np.sqrt(mse)
            }

        except Exception as e:
            self.logger.error(f"Error training CPU model: {e}")
            return {"success": False, "error": str(e)}

    async def _train_memory_model(self) -> Dict[str, Any]:
        """Train memory usage prediction model"""
        try:
            if not SKLEARN_AVAILABLE:
                return {"success": False, "error": "scikit-learn not available"}

            # Similar to CPU model training
            data = self.training_data["memory_usage"]
            if len(data) < 50:
                return {"success": False, "error": "Insufficient training data"}

            X = [d["features"] for d in data[:-1]]
            y = [d["value"] for d in data[1:]]

            X_scaled = self.scalers["memory_scaler"].fit_transform(X)
            self.models["memory_predictor"].fit(X_scaled, y)

            y_pred = self.models["memory_predictor"].predict(X_scaled)
            mse = mean_squared_error(y, y_pred)

            return {
                "success": True,
                "samples_used": len(X),
                "mse": mse,
                "rmse": np.sqrt(mse)
            }

        except Exception as e:
            self.logger.error(f"Error training memory model: {e}")
            return {"success": False, "error": str(e)}

    async def _train_command_model(self) -> Dict[str, Any]:
        """Train command frequency prediction model"""
        try:
            if not SKLEARN_AVAILABLE:
                return {"success": False, "error": "scikit-learn not available"}

            # Similar training for command frequency
            data = self.training_data["command_frequency"]
            if len(data) < 50:
                return {"success": False, "error": "Insufficient training data"}

            X = [d["features"] for d in data[:-1]]
            y = [d["value"] for d in data[1:]]

            X_scaled = self.scalers["command_scaler"].fit_transform(X)
            self.models["command_predictor"].fit(X_scaled, y)

            y_pred = self.models["command_predictor"].predict(X_scaled)
            mse = mean_squared_error(y, y_pred)

            return {
                "success": True,
                "samples_used": len(X),
                "mse": mse,
                "rmse": np.sqrt(mse)
            }

        except Exception as e:
            self.logger.error(f"Error training command model: {e}")
            return {"success": False, "error": str(e)}

    async def predict_cpu_usage(self, hours_ahead: int = 1) -> Dict[str, Any]:
        """Predict CPU usage"""
        try:
            current_time = time.time()

            # Extract current features
            features = self._extract_features(current_time, "cpu")

            if SKLEARN_AVAILABLE and "cpu_predictor" in self.models:
                # Use scikit-learn model
                features_scaled = self.scalers["cpu_scaler"].transform([features])
                prediction = self.models["cpu_predictor"].predict(features_scaled)[0]

                return {
                    "success": True,
                    "predicted_cpu_percent": max(0, min(100, prediction)),
                    "hours_ahead": hours_ahead,
                    "model": "scikit-learn",
                    "confidence": 0.8
                }

            else:
                # Simple heuristic prediction
                if self.training_data["cpu_usage"]:
                    recent_avg = np.mean([d["value"] for d in self.training_data["cpu_usage"][-10:]])
                    return {
                        "success": True,
                        "predicted_cpu_percent": recent_avg,
                        "hours_ahead": hours_ahead,
                        "model": "heuristic",
                        "confidence": 0.6
                    }

        except Exception as e:
            self.logger.error(f"Error predicting CPU usage: {e}")
            return {"success": False, "error": str(e)}

    async def predict_memory_usage(self, hours_ahead: int = 1) -> Dict[str, Any]:
        """Predict memory usage"""
        try:
            current_time = time.time()
            features = self._extract_features(current_time, "memory")

            if SKLEARN_AVAILABLE and "memory_predictor" in self.models:
                features_scaled = self.scalers["memory_scaler"].transform([features])
                prediction = self.models["memory_predictor"].predict(features_scaled)[0]

                return {
                    "success": True,
                    "predicted_memory_percent": max(0, min(100, prediction)),
                    "hours_ahead": hours_ahead,
                    "model": "scikit-learn",
                    "confidence": 0.8
                }

            else:
                # Heuristic prediction
                if self.training_data["memory_usage"]:
                    recent_avg = np.mean([d["value"] for d in self.training_data["memory_usage"][-10:]])
                    return {
                        "success": True,
                        "predicted_memory_percent": recent_avg,
                        "hours_ahead": hours_ahead,
                        "model": "heuristic",
                        "confidence": 0.6
                    }

        except Exception as e:
            self.logger.error(f"Error predicting memory usage: {e}")
            return {"success": False, "error": str(e)}

    async def predict_user_activity(self, hours_ahead: int = 1) -> Dict[str, Any]:
        """Predict user activity levels"""
        try:
            current_time = time.time()
            features = self._extract_features(current_time, "commands")

            if SKLEARN_AVAILABLE and "command_predictor" in self.models:
                features_scaled = self.scalers["command_scaler"].transform([features])
                prediction = self.models["command_predictor"].predict(features_scaled)[0]

                return {
                    "success": True,
                    "predicted_commands": max(0, prediction),
                    "hours_ahead": hours_ahead,
                    "model": "scikit-learn",
                    "confidence": 0.7
                }

            else:
                # Heuristic prediction based on time of day
                hour = time.localtime(current_time).tm_hour

                # Simple activity pattern (higher during work hours)
                if 9 <= hour <= 17:
                    predicted_commands = 50
                elif 18 <= hour <= 22:
                    predicted_commands = 20
                else:
                    predicted_commands = 5

                return {
                    "success": True,
                    "predicted_commands": predicted_commands,
                    "hours_ahead": hours_ahead,
                    "model": "time-based",
                    "confidence": 0.5
                }

        except Exception as e:
            self.logger.error(f"Error predicting user activity: {e}")
            return {"success": False, "error": str(e)}

    async def predict_system_issues(self) -> Dict[str, Any]:
        """Predict potential system issues"""
        try:
            issues = []

            # Analyze trends
            if len(self.training_data["cpu_usage"]) >= 20:
                recent_cpu = [d["value"] for d in self.training_data["cpu_usage"][-20:]]
                cpu_trend = np.polyfit(range(len(recent_cpu)), recent_cpu, 1)[0]

                if cpu_trend > 5:  # Increasing trend
                    issues.append({
                        "type": "cpu_trend",
                        "severity": "medium",
                        "description": "CPU usage trending upward",
                        "probability": min(0.9, cpu_trend / 10)
                    })

            if len(self.training_data["memory_usage"]) >= 20:
                recent_memory = [d["value"] for d in self.training_data["memory_usage"][-20:]]
                memory_trend = np.polyfit(range(len(recent_memory)), recent_memory, 1)[0]

                if memory_trend > 3:  # Increasing trend
                    issues.append({
                        "type": "memory_trend",
                        "severity": "high",
                        "description": "Memory usage trending upward",
                        "probability": min(0.9, memory_trend / 5)
                    })

            return {
                "success": True,
                "predicted_issues": issues,
                "total_issues": len(issues),
                "risk_level": "high" if len(issues) >= 3 else "medium" if len(issues) >= 1 else "low"
            }

        except Exception as e:
            self.logger.error(f"Error predicting system issues: {e}")
            return {"success": False, "error": str(e)}

    async def _prediction_update_loop(self):
        """Background loop for updating predictions"""
        while True:
            try:
                await asyncio.sleep(self.config["training_interval"])

                # Collect new training data
                await self.collect_training_data()

                # Update predictions
                await self._update_predictions()

                # Retrain models periodically
                if time.time() % self.config["model_update_frequency"] < self.config["training_interval"]:
                    await self.train_models()

            except Exception as e:
                self.logger.error(f"Error in prediction update loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry

    async def _update_predictions(self):
        """Update current predictions"""
        try:
            # Update CPU prediction
            cpu_prediction = await self.predict_cpu_usage(1)
            if cpu_prediction["success"]:
                self.predictions["cpu_usage_1h"] = cpu_prediction

            # Update memory prediction
            memory_prediction = await self.predict_memory_usage(1)
            if memory_prediction["success"]:
                self.predictions["memory_usage_1h"] = memory_prediction

            # Update user activity prediction
            activity_prediction = await self.predict_user_activity(1)
            if activity_prediction["success"]:
                self.predictions["user_activity_1h"] = activity_prediction

            # Update system issues prediction
            issues_prediction = await self.predict_system_issues()
            if issues_prediction["success"]:
                self.predictions["system_issues"] = issues_prediction

            self.stats["predictions_made"] += 1

        except Exception as e:
            self.logger.error(f"Error updating predictions: {e}")

    def get_current_predictions(self) -> Dict[str, Any]:
        """Get current predictions"""
        return {
            "predictions": self.predictions,
            "last_updated": time.time(),
            "ml_enabled": self.config["enable_ml_predictions"],
            "training_data_size": sum(len(data) for data in self.training_data.values())
        }

    def get_prediction_accuracy(self) -> Dict[str, Any]:
        """Get prediction accuracy metrics"""
        return {
            "total_predictions": self.stats["predictions_made"],
            "models_trained": self.stats["models_trained"],
            "training_cycles": self.stats["training_cycles"],
            "ml_libraries_available": {
                "scikit_learn": SKLEARN_AVAILABLE,
                "tensorflow": TENSORFLOW_AVAILABLE,
                "pytorch": PYTORCH_AVAILABLE
            }
        }

    async def save_training_data(self):
        """Save training data to disk"""
        try:
            data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_data.json')

            save_data = {
                "training_data": self.training_data,
                "last_saved": time.time(),
                "stats": self.stats
            }

            os.makedirs(os.path.dirname(data_file), exist_ok=True)
            with open(data_file, 'w') as f:
                import json
                json.dump(save_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving training data: {e}")

    async def shutdown(self):
        """Shutdown predictive analytics"""
        try:
            self.logger.info("Shutting down predictive analytics...")

            # Save training data
            await self.save_training_data()

            self.logger.info("Predictive analytics shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down predictive analytics: {e}")