"""
J.A.R.V.I.S. Advanced Predictive Analytics System
Machine learning-based prediction of system issues and failures
"""

import os
import time
import json
import logging
import asyncio
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics
import psutil
from collections import deque


@dataclass
class PredictionModel:
    """Machine learning model for predictions"""
    name: str
    type: str  # regression, classification, time_series
    features: List[str]
    target: str
    accuracy: float = 0.0
    last_trained: float = 0.0
    predictions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SystemMetrics:
    """System metrics for prediction"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    thread_count: int
    temperature: Optional[float] = None


class TimeSeriesAnalyzer:
    """Analyzes time series data for trend prediction"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history: deque[SystemMetrics] = deque(maxlen=window_size)

    def add_metrics(self, metrics: SystemMetrics):
        """Add new metrics to history"""
        self.metrics_history.append(metrics)

    def predict_trend(self, metric_name: str, hours_ahead: int = 24) -> Dict[str, Any]:
        """Predict future values using trend analysis"""
        if len(self.metrics_history) < 10:
            return {"error": "Insufficient data for prediction"}

        # Extract metric values
        values = []
        timestamps = []

        for metric in self.metrics_history:
            value = getattr(metric, metric_name, None)
            if value is not None:
                values.append(value)
                timestamps.append(metric.timestamp)

        if len(values) < 5:
            return {"error": "Insufficient metric data"}

        # Calculate trend
        try:
            slope = self._calculate_trend_slope(values, timestamps)
            intercept = statistics.mean(values) - slope * statistics.mean(timestamps)

            # Predict future value
            future_timestamp = timestamps[-1] + (hours_ahead * 3600)
            predicted_value = slope * future_timestamp + intercept

            # Calculate confidence
            r_squared = self._calculate_r_squared(values, timestamps, slope, intercept)

            # Determine if trend indicates problem
            current_avg = statistics.mean(values[-10:])  # Last 10 readings
            threshold_breached = self._check_threshold_breach(metric_name, predicted_value)

            return {
                "predicted_value": predicted_value,
                "confidence": r_squared,
                "trend": "increasing" if slope > 0 else "decreasing",
                "slope": slope,
                "hours_ahead": hours_ahead,
                "threshold_breached": threshold_breached,
                "severity": self._assess_severity(metric_name, predicted_value, current_avg)
            }

        except Exception as e:
            return {"error": str(e)}

    def _calculate_trend_slope(self, values: List[float], timestamps: List[float]) -> float:
        """Calculate linear regression slope"""
        n = len(values)
        sum_x = sum(timestamps)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(timestamps, values))
        sum_x2 = sum(x * x for x in timestamps)

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0

        return (n * sum_xy - sum_x * sum_y) / denominator

    def _calculate_r_squared(self, values: List[float], timestamps: List[float],
                           slope: float, intercept: float) -> float:
        """Calculate R-squared for trend line fit"""
        actual_mean = statistics.mean(values)

        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(timestamps, values))
        ss_tot = sum((y - actual_mean) ** 2 for y in values)

        if ss_tot == 0:
            return 1.0

        return 1 - (ss_res / ss_tot)

    def _check_threshold_breach(self, metric_name: str, predicted_value: float) -> bool:
        """Check if predicted value breaches thresholds"""
        thresholds = {
            "cpu_percent": 80,
            "memory_percent": 85,
            "disk_percent": 90
        }

        threshold = thresholds.get(metric_name, 100)
        return predicted_value > threshold

    def _assess_severity(self, metric_name: str, predicted_value: float, current_avg: float) -> str:
        """Assess severity of predicted issue"""
        if metric_name in ["cpu_percent", "memory_percent", "disk_percent"]:
            if predicted_value > 95:
                return "critical"
            elif predicted_value > 85:
                return "high"
            elif predicted_value > 75:
                return "medium"
            else:
                return "low"
        return "unknown"


class FailurePredictor:
    """Predicts system failures using historical data"""

    def __init__(self):
        self.failure_history = []
        self.patterns = {}

    def analyze_failure_patterns(self) -> Dict[str, Any]:
        """Analyze historical failure patterns"""
        if not self.failure_history:
            return {"patterns": [], "insights": []}

        # Group failures by type
        failure_types = {}
        for failure in self.failure_history:
            f_type = failure.get("type", "unknown")
            if f_type not in failure_types:
                failure_types[f_type] = []
            failure_types[f_type].append(failure)

        patterns = []
        for f_type, failures in failure_types.items():
            if len(failures) >= 3:  # Need at least 3 occurrences for pattern
                pattern = self._extract_pattern(f_type, failures)
                if pattern:
                    patterns.append(pattern)

        return {
            "patterns": patterns,
            "total_failures": len(self.failure_history),
            "unique_failure_types": len(failure_types),
            "insights": self._generate_insights(patterns)
        }

    def _extract_pattern(self, failure_type: str, failures: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract pattern from failure data"""
        if len(failures) < 3:
            return None

        # Calculate time intervals between failures
        timestamps = [f.get("timestamp", 0) for f in failures]
        timestamps.sort()

        intervals = []
        for i in range(1, len(timestamps)):
            intervals.append(timestamps[i] - timestamps[i-1])

        if intervals:
            avg_interval = statistics.mean(intervals)
            std_dev = statistics.stdev(intervals) if len(intervals) > 1 else 0

            # Predict next occurrence
            next_occurrence = timestamps[-1] + avg_interval

            return {
                "failure_type": failure_type,
                "frequency": len(failures),
                "average_interval_hours": avg_interval / 3600,
                "variability": std_dev / avg_interval if avg_interval > 0 else 0,
                "next_predicted": next_occurrence,
                "confidence": min(1.0, len(failures) / 10)  # Higher confidence with more data
            }

        return None

    def _generate_insights(self, patterns: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from failure patterns"""
        insights = []

        for pattern in patterns:
            freq = pattern.get("frequency", 0)
            interval = pattern.get("average_interval_hours", 0)
            variability = pattern.get("variability", 0)

            if freq >= 5:
                insights.append(f"High frequency {pattern['failure_type']} failures ({freq} occurrences)")
            elif interval < 24:  # Less than daily
                insights.append(f"Frequent {pattern['failure_type']} failures (every {interval:.1f} hours)")
            elif variability > 0.5:
                insights.append(f"Unpredictable {pattern['failure_type']} failure timing")

        return insights

    def predict_next_failure(self, failure_type: str) -> Dict[str, Any]:
        """Predict when next failure of given type will occur"""
        relevant_failures = [f for f in self.failure_history if f.get("type") == failure_type]

        if len(relevant_failures) < 3:
            return {"error": "Insufficient data for prediction"}

        pattern = self._extract_pattern(failure_type, relevant_failures)

        if pattern:
            next_time = pattern["next_predicted"]
            confidence = pattern["confidence"]

            return {
                "failure_type": failure_type,
                "next_occurrence": next_time,
                "confidence": confidence,
                "time_until": next_time - time.time(),
                "hours_until": (next_time - time.time()) / 3600
            }

        return {"error": "Could not extract failure pattern"}


class AnomalyDetector:
    """Detects anomalies in system behavior"""

    def __init__(self):
        self.baseline_metrics = {}
        self.anomaly_history = []

    def establish_baseline(self, metrics_history: List[SystemMetrics], window_days: int = 7):
        """Establish baseline metrics from historical data"""
        if len(metrics_history) < 10:
            return {"error": "Insufficient data for baseline"}

        # Calculate baselines for each metric
        baselines = {}

        # CPU baseline
        cpu_values = [m.cpu_percent for m in metrics_history]
        baselines["cpu_percent"] = {
            "mean": statistics.mean(cpu_values),
            "std_dev": statistics.stdev(cpu_values),
            "min": min(cpu_values),
            "max": max(cpu_values)
        }

        # Memory baseline
        memory_values = [m.memory_percent for m in metrics_history]
        baselines["memory_percent"] = {
            "mean": statistics.mean(memory_values),
            "std_dev": statistics.stdev(memory_values),
            "min": min(memory_values),
            "max": max(memory_values)
        }

        # Disk baseline
        disk_values = [m.disk_percent for m in metrics_history]
        baselines["disk_percent"] = {
            "mean": statistics.mean(disk_values),
            "std_dev": statistics.stdev(disk_values),
            "min": min(disk_values),
            "max": max(disk_values)
        }

        self.baseline_metrics = baselines
        return {"success": True, "baselines": baselines}

    def detect_anomalies(self, current_metrics: SystemMetrics) -> List[Dict[str, Any]]:
        """Detect anomalies in current metrics"""
        anomalies = []

        if not self.baseline_metrics:
            return anomalies

        # Check each metric
        metrics_to_check = [
            ("cpu_percent", current_metrics.cpu_percent),
            ("memory_percent", current_metrics.memory_percent),
            ("disk_percent", current_metrics.disk_percent)
        ]

        for metric_name, current_value in metrics_to_check:
            if metric_name in self.baseline_metrics:
                baseline = self.baseline_metrics[metric_name]
                anomaly = self._check_metric_anomaly(metric_name, current_value, baseline)

                if anomaly:
                    anomalies.append(anomaly)

        return anomalies

    def _check_metric_anomaly(self, metric_name: str, current_value: float,
                            baseline: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if a metric value is anomalous"""
        mean = baseline["mean"]
        std_dev = baseline["std_dev"]

        if std_dev == 0:
            # No variation in baseline, check against mean
            deviation = abs(current_value - mean)
            threshold = max(mean * 0.2, 5)  # 20% or 5 units
        else:
            # Use standard deviation
            deviation = abs(current_value - mean)
            threshold = 3 * std_dev  # 3-sigma rule

        if deviation > threshold:
            severity = "high" if deviation > threshold * 2 else "medium"

            return {
                "metric": metric_name,
                "current_value": current_value,
                "expected_range": f"{mean - threshold:.1f} - {mean + threshold:.1f}",
                "deviation": deviation,
                "severity": severity,
                "timestamp": time.time()
            }

        return None


class Predictor:
    """Advanced predictive analytics system for system issues"""

    def __init__(self, healer):
        self.healer = healer
        self.jarvis = healer.jarvis
        self.logger = logging.getLogger('JARVIS.Predictor')

        # Prediction components
        self.time_series_analyzer = TimeSeriesAnalyzer()
        self.failure_predictor = FailurePredictor()
        self.anomaly_detector = AnomalyDetector()

        # Models and data
        self.prediction_models: Dict[str, PredictionModel] = {}
        self.metrics_history: List[SystemMetrics] = []
        self.prediction_history: List[Dict[str, Any]] = []

        # Configuration
        self.monitoring_interval = 60  # 1 minute
        self.prediction_horizon_hours = 24
        self.baseline_window_days = 7

    async def initialize(self):
        """Initialize the advanced predictor"""
        try:
            self.logger.info("Initializing advanced predictive analytics system...")

            # Load historical data
            await self._load_historical_data()

            # Establish baselines
            if len(self.metrics_history) >= 10:
                self.anomaly_detector.establish_baseline(self.metrics_history, self.baseline_window_days)

            # Start monitoring
            asyncio.create_task(self._monitoring_loop())

            self.logger.info("Advanced predictor initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing predictor: {e}")
            raise

    async def predict_issues(self, prediction_horizon: int = None) -> Dict[str, Any]:
        """Generate comprehensive issue predictions"""
        horizon = prediction_horizon or self.prediction_horizon_hours

        predictions = {
            "timestamp": time.time(),
            "horizon_hours": horizon,
            "predictions": [],
            "anomalies": [],
            "failure_patterns": [],
            "confidence": 0.0,
            "risk_assessment": "low"
        }

        try:
            # Trend-based predictions
            trend_predictions = await self._generate_trend_predictions(horizon)
            predictions["predictions"].extend(trend_predictions)

            # Anomaly detection
            current_metrics = self._collect_current_metrics()
            anomalies = self.anomaly_detector.detect_anomalies(current_metrics)
            predictions["anomalies"] = anomalies

            # Failure pattern analysis
            failure_patterns = self.failure_predictor.analyze_failure_patterns()
            predictions["failure_patterns"] = failure_patterns.get("patterns", [])

            # Calculate overall confidence and risk
            predictions["confidence"] = self._calculate_prediction_confidence(
                trend_predictions, anomalies, failure_patterns
            )

            predictions["risk_assessment"] = self._assess_risk_level(
                trend_predictions, anomalies, failure_patterns
            )

            # Store prediction
            self.prediction_history.append(predictions)

            # Cleanup old predictions
            if len(self.prediction_history) > 100:
                self.prediction_history.pop(0)

        except Exception as e:
            self.logger.error(f"Error generating predictions: {e}")
            predictions["error"] = str(e)

        return predictions

    async def _generate_trend_predictions(self, horizon: int) -> List[Dict[str, Any]]:
        """Generate predictions based on trend analysis"""
        predictions = []

        metrics_to_predict = ["cpu_percent", "memory_percent", "disk_percent"]

        for metric in metrics_to_predict:
            prediction = self.time_series_analyzer.predict_trend(metric, horizon)

            if "error" not in prediction:
                predictions.append({
                    "type": "trend_prediction",
                    "metric": metric,
                    "predicted_value": prediction["predicted_value"],
                    "confidence": prediction["confidence"],
                    "trend": prediction["trend"],
                    "threshold_breached": prediction.get("threshold_breached", False),
                    "severity": prediction.get("severity", "low"),
                    "timeframe_hours": horizon
                })

        return predictions

    def _collect_current_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            process_count = len(psutil.pids())

            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                network_bytes_sent=network.bytes_sent if network else 0,
                network_bytes_recv=network.bytes_recv if network else 0,
                process_count=process_count,
                thread_count=threading.active_count()
            )

        except Exception as e:
            self.logger.debug(f"Error collecting metrics: {e}")
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=0,
                memory_percent=0,
                disk_percent=0,
                network_bytes_sent=0,
                network_bytes_recv=0,
                process_count=0,
                thread_count=0
            )

    def _calculate_prediction_confidence(self, trend_predictions: List[Dict[str, Any]],
                                       anomalies: List[Dict[str, Any]],
                                       failure_patterns: Dict[str, Any]) -> float:
        """Calculate overall prediction confidence"""
        confidence_factors = []

        # Trend prediction confidence
        if trend_predictions:
            avg_trend_confidence = statistics.mean(p["confidence"] for p in trend_predictions)
            confidence_factors.append(avg_trend_confidence)

        # Anomaly detection confidence (higher if anomalies detected)
        anomaly_confidence = min(0.9, len(anomalies) * 0.1 + 0.5)
        confidence_factors.append(anomaly_confidence)

        # Failure pattern confidence
        pattern_confidence = min(0.8, len(failure_patterns.get("patterns", [])) * 0.1 + 0.4)
        confidence_factors.append(pattern_confidence)

        # Historical data confidence
        data_confidence = min(1.0, len(self.metrics_history) / 1000)
        confidence_factors.append(data_confidence)

        return statistics.mean(confidence_factors) if confidence_factors else 0.0

    def _assess_risk_level(self, trend_predictions: List[Dict[str, Any]],
                          anomalies: List[Dict[str, Any]],
                          failure_patterns: Dict[str, Any]) -> str:
        """Assess overall risk level"""
        risk_score = 0

        # Trend-based risk
        for prediction in trend_predictions:
            if prediction.get("threshold_breached", False):
                severity = prediction.get("severity", "low")
                if severity == "critical":
                    risk_score += 5
                elif severity == "high":
                    risk_score += 3
                elif severity == "medium":
                    risk_score += 1

        # Anomaly-based risk
        risk_score += len(anomalies) * 2

        # Failure pattern risk
        risk_score += len(failure_patterns.get("patterns", [])) * 1.5

        if risk_score >= 10:
            return "critical"
        elif risk_score >= 7:
            return "high"
        elif risk_score >= 4:
            return "medium"
        elif risk_score >= 2:
            return "low"
        else:
            return "minimal"

    async def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while True:
            try:
                # Collect metrics
                metrics = self._collect_current_metrics()
                self.metrics_history.append(metrics)
                self.time_series_analyzer.add_metrics(metrics)

                # Keep history manageable
                if len(self.metrics_history) > 10000:  # Keep last 10000 readings
                    self.metrics_history = self.metrics_history[-10000:]

                # Check for anomalies
                anomalies = self.anomaly_detector.detect_anomalies(metrics)
                if anomalies:
                    self.logger.warning(f"Detected {len(anomalies)} system anomalies")

                    # Could trigger alerts here
                    if hasattr(self.jarvis, 'event_manager'):
                        for anomaly in anomalies:
                            await self.jarvis.event_manager.emit_event(
                                "system_anomaly_detected",
                                anomaly
                            )

                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retry

    async def _load_historical_data(self):
        """Load historical metrics and prediction data"""
        try:
            # Load metrics history
            metrics_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'metrics_history.json')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)

                # Convert back to SystemMetrics objects
                for item in metrics_data.get("history", []):
                    metric = SystemMetrics(**item)
                    self.metrics_history.append(metric)
                    self.time_series_analyzer.add_metrics(metric)

            # Load prediction history
            predictions_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'predictions_history.json')
            if os.path.exists(predictions_file):
                with open(predictions_file, 'r') as f:
                    self.prediction_history = json.load(f).get("history", [])

        except Exception as e:
            self.logger.debug(f"Could not load historical data: {e}")

    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction statistics"""
        if not self.prediction_history:
            return {"total_predictions": 0}

        total_predictions = len(self.prediction_history)
        avg_confidence = statistics.mean(p.get("confidence", 0) for p in self.prediction_history)

        risk_levels = [p.get("risk_assessment", "unknown") for p in self.prediction_history]
        risk_distribution = {
            "critical": risk_levels.count("critical"),
            "high": risk_levels.count("high"),
            "medium": risk_levels.count("medium"),
            "low": risk_levels.count("low"),
            "minimal": risk_levels.count("minimal")
        }

        return {
            "total_predictions": total_predictions,
            "average_confidence": avg_confidence,
            "risk_distribution": risk_distribution,
            "metrics_collected": len(self.metrics_history),
            "anomalies_detected": sum(len(p.get("anomalies", [])) for p in self.prediction_history)
        }

    async def shutdown(self):
        """Shutdown the advanced predictor"""
        try:
            self.logger.info("Shutting down advanced predictor...")

            # Save data
            await self._save_historical_data()

            self.logger.info("Advanced predictor shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down predictor: {e}")

    async def _save_historical_data(self):
        """Save historical data"""
        try:
            # Save metrics history
            metrics_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'metrics_history.json')
            os.makedirs(os.path.dirname(metrics_file), exist_ok=True)

            metrics_data = {
                "history": [
                    {
                        "timestamp": m.timestamp,
                        "cpu_percent": m.cpu_percent,
                        "memory_percent": m.memory_percent,
                        "disk_percent": m.disk_percent,
                        "network_bytes_sent": m.network_bytes_sent,
                        "network_bytes_recv": m.network_bytes_recv,
                        "process_count": m.process_count,
                        "thread_count": m.thread_count,
                        "temperature": m.temperature
                    }
                    for m in self.metrics_history[-1000:]  # Last 1000 entries
                ],
                "last_updated": time.time()
            }

            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)

            # Save predictions history
            predictions_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'predictions_history.json')

            predictions_data = {
                "history": self.prediction_history[-100:],  # Last 100 predictions
                "stats": self.get_prediction_stats(),
                "last_updated": time.time()
            }

            with open(predictions_file, 'w') as f:
                json.dump(predictions_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving historical data: {e}")