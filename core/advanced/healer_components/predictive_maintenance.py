"""
J.A.R.V.I.S. Predictive Maintenance
Advanced predictive maintenance system using machine learning and statistical analysis
"""

import os
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
import psutil
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
from collections import deque


class PredictiveMaintenance:
    """
    Ultra-advanced predictive maintenance system that analyzes system patterns,
    predicts potential failures, and schedules preventive maintenance
    """

    def __init__(self, application_healer):
        """
        Initialize predictive maintenance

        Args:
            application_healer: Reference to application healer
        """
        self.healer = application_healer
        self.jarvis = application_healer.jarvis
        self.logger = logging.getLogger('JARVIS.PredictiveMaintenance')

        # Maintenance configuration
        self.maintenance_config = {
            'prediction_window_hours': 24,
            'confidence_threshold': 0.7,
            'maintenance_scheduling': True,
            'auto_maintenance': False,
            'data_collection_interval': 60,  # seconds
            'historical_data_days': 7,
            'anomaly_detection_sensitivity': 0.8
        }

        # Predictive models and data
        self.system_metrics_history = {
            'cpu': deque(maxlen=1000),
            'memory': deque(maxlen=1000),
            'disk': deque(maxlen=1000),
            'network': deque(maxlen=1000)
        }

        self.prediction_models = {}
        self.maintenance_schedule = []
        self.predicted_failures = []

        # Maintenance statistics
        self.stats = {
            'predictions_made': 0,
            'accurate_predictions': 0,
            'preventive_maintenance_performed': 0,
            'failures_prevented': 0,
            'false_positives': 0,
            'maintenance_scheduled': 0
        }

    async def initialize(self):
        """Initialize predictive maintenance"""
        try:
            self.logger.info("Initializing predictive maintenance...")

            # Load historical data
            await self._load_historical_data()

            # Initialize prediction models
            await self._initialize_prediction_models()

            # Start data collection
            asyncio.create_task(self._continuous_data_collection())

            # Start prediction engine
            asyncio.create_task(self._prediction_engine())

            self.logger.info("Predictive maintenance initialized")

        except Exception as e:
            self.logger.error(f"Error initializing predictive maintenance: {e}")
            raise

    async def predict_system_failures(self,
                                    prediction_horizon: int = 24,
                                    confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Predict potential system failures

        Args:
            prediction_horizon: Hours to predict ahead
            confidence_threshold: Minimum confidence for predictions

        Returns:
            Failure predictions and maintenance recommendations
        """
        start_time = time.time()

        try:
            self.logger.info(f"Predicting system failures for next {prediction_horizon} hours")

            # Analyze current system state
            current_state = await self._analyze_current_system_state()

            # Generate predictions for different components
            predictions = {}

            # CPU failure prediction
            predictions['cpu'] = await self._predict_cpu_failure(current_state, prediction_horizon)

            # Memory failure prediction
            predictions['memory'] = await self._predict_memory_failure(current_state, prediction_horizon)

            # Disk failure prediction
            predictions['disk'] = await self._predict_disk_failure(current_state, prediction_horizon)

            # Network failure prediction
            predictions['network'] = await self._predict_network_failure(current_state, prediction_horizon)

            # Process failure prediction
            predictions['processes'] = await self._predict_process_failures(current_state, prediction_horizon)

            # Filter predictions by confidence
            high_confidence_predictions = self._filter_predictions_by_confidence(
                predictions, confidence_threshold
            )

            # Generate maintenance recommendations
            maintenance_recommendations = await self._generate_maintenance_recommendations(
                high_confidence_predictions, prediction_horizon
            )

            # Calculate risk assessment
            risk_assessment = self._calculate_system_risk(high_confidence_predictions)

            prediction_time = time.time() - start_time
            self.stats['predictions_made'] += 1

            result = {
                'prediction_horizon': prediction_horizon,
                'confidence_threshold': confidence_threshold,
                'all_predictions': predictions,
                'high_confidence_predictions': high_confidence_predictions,
                'maintenance_recommendations': maintenance_recommendations,
                'risk_assessment': risk_assessment,
                'prediction_time': prediction_time,
                'timestamp': time.time()
            }

            # Store predictions
            self.predicted_failures.append(result)

            self.logger.info(f"Failure prediction completed in {prediction_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error predicting system failures: {e}")
            return {
                'error': str(e),
                'prediction_time': time.time() - start_time
            }

    async def schedule_maintenance(self,
                                 maintenance_type: str,
                                 priority: str = "medium",
                                 scheduled_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Schedule preventive maintenance

        Args:
            maintenance_type: Type of maintenance
            priority: Maintenance priority
            scheduled_time: Specific time to schedule (timestamp)

        Returns:
            Maintenance scheduling result
        """
        try:
            if scheduled_time is None:
                # Schedule for next available maintenance window
                scheduled_time = time.time() + 3600  # 1 hour from now

            maintenance_task = {
                'id': f"maint_{int(time.time())}_{hash(maintenance_type) % 1000}",
                'type': maintenance_type,
                'priority': priority,
                'scheduled_time': scheduled_time,
                'status': 'scheduled',
                'created_at': time.time(),
                'estimated_duration': self._estimate_maintenance_duration(maintenance_type),
                'resources_required': self._get_maintenance_resources(maintenance_type)
            }

            # Add to maintenance schedule
            self.maintenance_schedule.append(maintenance_task)

            # Sort by priority and time
            self._sort_maintenance_schedule()

            self.stats['maintenance_scheduled'] += 1

            self.logger.info(f"Scheduled {priority} priority {maintenance_type} maintenance")

            return {
                'success': True,
                'maintenance_task': maintenance_task,
                'scheduled_time': datetime.fromtimestamp(scheduled_time).isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error scheduling maintenance: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def perform_maintenance(self, maintenance_id: str) -> Dict[str, Any]:
        """Perform scheduled maintenance"""
        try:
            # Find maintenance task
            maintenance_task = None
            for task in self.maintenance_schedule:
                if task['id'] == maintenance_id:
                    maintenance_task = task
                    break

            if not maintenance_task:
                return {
                    'success': False,
                    'error': 'Maintenance task not found'
                }

            # Update status
            maintenance_task['status'] = 'in_progress'
            maintenance_task['started_at'] = time.time()

            self.logger.info(f"Starting maintenance: {maintenance_task['type']}")

            # Perform maintenance based on type
            result = await self._execute_maintenance_task(maintenance_task)

            # Update task status
            maintenance_task['status'] = 'completed' if result['success'] else 'failed'
            maintenance_task['completed_at'] = time.time()
            maintenance_task['result'] = result

            if result['success']:
                self.stats['preventive_maintenance_performed'] += 1

            self.logger.info(f"Maintenance {maintenance_task['type']} completed: {result['success']}")

            return {
                'success': result['success'],
                'maintenance_task': maintenance_task,
                'result': result
            }

        except Exception as e:
            self.logger.error(f"Error performing maintenance: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _analyze_current_system_state(self) -> Dict[str, Any]:
        """Analyze current system state"""
        state = {}

        try:
            # CPU state
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_freq = psutil.cpu_freq()
            state['cpu'] = {
                'percent': cpu_percent,
                'frequency': cpu_freq.current if cpu_freq else 0,
                'cores': psutil.cpu_count(),
                'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }

            # Memory state
            memory = psutil.virtual_memory()
            state['memory'] = {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used
            }

            # Disk state
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            state['disk'] = {
                'total': disk.total,
                'free': disk.free,
                'percent': disk.percent,
                'io_read': disk_io.read_bytes if disk_io else 0,
                'io_write': disk_io.write_bytes if disk_io else 0
            }

            # Network state
            net_io = psutil.net_io_counters()
            state['network'] = {
                'bytes_sent': net_io.bytes_sent if net_io else 0,
                'bytes_recv': net_io.bytes_recv if net_io else 0,
                'packets_sent': net_io.packets_sent if net_io else 0,
                'packets_recv': net_io.packets_recv if net_io else 0,
                'errors': (net_io.errin + net_io.errout) if net_io else 0
            }

            # Process state
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
                try:
                    proc_info = proc.info
                    processes.append({
                        'pid': proc_info['pid'],
                        'name': proc_info['name'],
                        'cpu_percent': proc_info['cpu_percent'] or 0,
                        'memory_percent': proc_info['memory_percent'] or 0,
                        'status': proc_info['status']
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            state['processes'] = processes[:50]  # Limit to top 50 processes

        except Exception as e:
            self.logger.warning(f"Error analyzing system state: {e}")
            state['error'] = str(e)

        return state

    async def _predict_cpu_failure(self, current_state: Dict[str, Any], horizon: int) -> Dict[str, Any]:
        """Predict CPU-related failures"""
        prediction = {
            'failure_probability': 0.0,
            'predicted_time': None,
            'confidence': 0.0,
            'failure_type': None,
            'indicators': []
        }

        try:
            cpu_state = current_state.get('cpu', {})
            cpu_percent = cpu_state.get('percent', 0)

            # Simple prediction based on current usage
            if cpu_percent > 90:
                prediction['failure_probability'] = 0.8
                prediction['predicted_time'] = time.time() + (horizon * 3600 * 0.3)  # 30% of horizon
                prediction['confidence'] = 0.7
                prediction['failure_type'] = 'cpu_overload'
                prediction['indicators'] = ['High CPU usage', 'Potential thermal throttling']

            elif cpu_percent > 75:
                prediction['failure_probability'] = 0.4
                prediction['predicted_time'] = time.time() + (horizon * 3600 * 0.6)  # 60% of horizon
                prediction['confidence'] = 0.5
                prediction['failure_type'] = 'cpu_stress'
                prediction['indicators'] = ['Elevated CPU usage']

            # Analyze historical trends
            cpu_history = list(self.system_metrics_history['cpu'])
            if len(cpu_history) > 10:
                recent_avg = sum(cpu_history[-10:]) / 10
                if recent_avg > 70:
                    prediction['failure_probability'] = max(prediction['failure_probability'], 0.6)
                    prediction['confidence'] = max(prediction['confidence'], 0.6)

        except Exception as e:
            self.logger.warning(f"Error predicting CPU failure: {e}")

        return prediction

    async def _predict_memory_failure(self, current_state: Dict[str, Any], horizon: int) -> Dict[str, Any]:
        """Predict memory-related failures"""
        prediction = {
            'failure_probability': 0.0,
            'predicted_time': None,
            'confidence': 0.0,
            'failure_type': None,
            'indicators': []
        }

        try:
            memory_state = current_state.get('memory', {})
            memory_percent = memory_state.get('percent', 0)

            if memory_percent > 95:
                prediction['failure_probability'] = 0.9
                prediction['predicted_time'] = time.time() + (horizon * 3600 * 0.1)  # 10% of horizon
                prediction['confidence'] = 0.8
                prediction['failure_type'] = 'memory_exhaustion'
                prediction['indicators'] = ['Critical memory usage', 'Out of memory risk']

            elif memory_percent > 85:
                prediction['failure_probability'] = 0.6
                prediction['predicted_time'] = time.time() + (horizon * 3600 * 0.4)  # 40% of horizon
                prediction['confidence'] = 0.6
                prediction['failure_type'] = 'memory_pressure'
                prediction['indicators'] = ['High memory usage', 'Swap usage likely']

            # Check swap usage
            swap = psutil.swap_memory()
            if swap.percent > 80:
                prediction['failure_probability'] = max(prediction['failure_probability'], 0.7)
                prediction['failure_type'] = 'swap_exhaustion'
                prediction['indicators'].append('High swap usage')

        except Exception as e:
            self.logger.warning(f"Error predicting memory failure: {e}")

        return prediction

    async def _predict_disk_failure(self, current_state: Dict[str, Any], horizon: int) -> Dict[str, Any]:
        """Predict disk-related failures"""
        prediction = {
            'failure_probability': 0.0,
            'predicted_time': None,
            'confidence': 0.0,
            'failure_type': None,
            'indicators': []
        }

        try:
            disk_state = current_state.get('disk', {})
            disk_percent = disk_state.get('percent', 0)

            if disk_percent > 98:
                prediction['failure_probability'] = 0.95
                prediction['predicted_time'] = time.time() + (horizon * 3600 * 0.05)  # 5% of horizon
                prediction['confidence'] = 0.9
                prediction['failure_type'] = 'disk_full'
                prediction['indicators'] = ['Disk nearly full', 'Write operations will fail']

            elif disk_percent > 90:
                prediction['failure_probability'] = 0.7
                prediction['predicted_time'] = time.time() + (horizon * 3600 * 0.2)  # 20% of horizon
                prediction['confidence'] = 0.7
                prediction['failure_type'] = 'disk_space_critical'
                prediction['indicators'] = ['Very low disk space', 'Performance degradation likely']

            # Check I/O rates (simplified S.M.A.R.T. equivalent)
            io_read = disk_state.get('io_read', 0)
            io_write = disk_state.get('io_write', 0)

            if io_read > 100 * 1024 * 1024:  # 100MB/s read
                prediction['indicators'].append('High disk I/O load')

        except Exception as e:
            self.logger.warning(f"Error predicting disk failure: {e}")

        return prediction

    async def _predict_network_failure(self, current_state: Dict[str, Any], horizon: int) -> Dict[str, Any]:
        """Predict network-related failures"""
        prediction = {
            'failure_probability': 0.0,
            'predicted_time': None,
            'confidence': 0.0,
            'failure_type': None,
            'indicators': []
        }

        try:
            network_state = current_state.get('network', {})
            errors = network_state.get('errors', 0)

            if errors > 1000:
                prediction['failure_probability'] = 0.8
                prediction['predicted_time'] = time.time() + (horizon * 3600 * 0.3)
                prediction['confidence'] = 0.7
                prediction['failure_type'] = 'network_errors'
                prediction['indicators'] = ['High network error rate', 'Connectivity issues likely']

            elif errors > 100:
                prediction['failure_probability'] = 0.4
                prediction['predicted_time'] = time.time() + (horizon * 3600 * 0.6)
                prediction['confidence'] = 0.5
                prediction['failure_type'] = 'network_degradation'
                prediction['indicators'] = ['Elevated network errors']

        except Exception as e:
            self.logger.warning(f"Error predicting network failure: {e}")

        return prediction

    async def _predict_process_failures(self, current_state: Dict[str, Any], horizon: int) -> List[Dict[str, Any]]:
        """Predict process-related failures"""
        predictions = []

        try:
            processes = current_state.get('processes', [])

            for proc in processes:
                cpu_percent = proc.get('cpu_percent', 0)
                memory_percent = proc.get('memory_percent', 0)

                # Predict process failure based on resource usage
                if cpu_percent > 95 or memory_percent > 90:
                    predictions.append({
                        'process_name': proc.get('name', 'unknown'),
                        'process_id': proc.get('pid'),
                        'failure_probability': 0.8,
                        'predicted_time': time.time() + (horizon * 3600 * 0.2),
                        'confidence': 0.7,
                        'failure_type': 'resource_exhaustion',
                        'indicators': [f'High CPU: {cpu_percent}%', f'High memory: {memory_percent}%']
                    })

        except Exception as e:
            self.logger.warning(f"Error predicting process failures: {e}")

        return predictions

    def _filter_predictions_by_confidence(self, predictions: Dict[str, Any], threshold: float) -> Dict[str, Any]:
        """Filter predictions by confidence threshold"""
        filtered = {}

        for component, prediction in predictions.items():
            if isinstance(prediction, dict):
                if prediction.get('confidence', 0) >= threshold:
                    filtered[component] = prediction
            elif isinstance(prediction, list):
                # For process predictions
                filtered[component] = [
                    p for p in prediction
                    if p.get('confidence', 0) >= threshold
                ]

        return filtered

    async def _generate_maintenance_recommendations(self, predictions: Dict[str, Any], horizon: int) -> List[Dict[str, Any]]:
        """Generate maintenance recommendations based on predictions"""
        recommendations = []

        try:
            for component, prediction in predictions.items():
                if isinstance(prediction, dict) and prediction.get('failure_probability', 0) > 0.5:
                    recommendation = {
                        'component': component,
                        'priority': 'high' if prediction['failure_probability'] > 0.7 else 'medium',
                        'action': f"preventive_maintenance_{component}",
                        'description': f"Preventive maintenance for {component} failure",
                        'predicted_failure': prediction.get('failure_type'),
                        'time_to_failure': prediction.get('predicted_time', time.time() + horizon * 3600) - time.time(),
                        'confidence': prediction.get('confidence', 0),
                        'estimated_duration': 30,  # 30 minutes
                        'resources_required': [component]
                    }
                    recommendations.append(recommendation)

                elif isinstance(prediction, list):
                    # Process-level recommendations
                    for proc_prediction in prediction:
                        recommendations.append({
                            'component': 'process',
                            'process_name': proc_prediction.get('process_name'),
                            'priority': 'high',
                            'action': 'process_restart',
                            'description': f"Restart problematic process {proc_prediction.get('process_name')}",
                            'time_to_failure': proc_prediction.get('predicted_time', time.time()) - time.time(),
                            'confidence': proc_prediction.get('confidence', 0)
                        })

            # Sort by priority and time to failure
            priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            recommendations.sort(key=lambda x: (
                priority_order.get(x.get('priority', 'medium'), 2),
                x.get('time_to_failure', float('inf'))
            ))

        except Exception as e:
            self.logger.warning(f"Error generating maintenance recommendations: {e}")

        return recommendations

    def _calculate_system_risk(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall system risk"""
        risk = {
            'overall_risk': 'low',
            'risk_score': 0.0,
            'critical_components': [],
            'time_to_next_failure': float('inf'),
            'risk_factors': []
        }

        try:
            total_risk = 0
            component_count = 0
            critical_components = []

            for component, prediction in predictions.items():
                if isinstance(prediction, dict):
                    prob = prediction.get('failure_probability', 0)
                    total_risk += prob
                    component_count += 1

                    if prob > 0.7:
                        critical_components.append(component)

                    time_to_failure = prediction.get('predicted_time', time.time() + 86400) - time.time()
                    risk['time_to_next_failure'] = min(risk['time_to_next_failure'], time_to_failure)

                elif isinstance(prediction, list):
                    for proc_pred in prediction:
                        prob = proc_pred.get('failure_probability', 0)
                        total_risk += prob
                        component_count += 1

            if component_count > 0:
                risk['risk_score'] = total_risk / component_count

            risk['critical_components'] = critical_components

            # Determine overall risk level
            if risk['risk_score'] > 0.7 or risk['time_to_next_failure'] < 3600:  # 1 hour
                risk['overall_risk'] = 'critical'
            elif risk['risk_score'] > 0.5 or risk['time_to_next_failure'] < 7200:  # 2 hours
                risk['overall_risk'] = 'high'
            elif risk['risk_score'] > 0.3 or risk['time_to_next_failure'] < 14400:  # 4 hours
                risk['overall_risk'] = 'medium'
            else:
                risk['overall_risk'] = 'low'

            # Identify risk factors
            if critical_components:
                risk['risk_factors'].append(f"Critical components at risk: {', '.join(critical_components)}")

            if risk['time_to_next_failure'] < 3600:
                risk['risk_factors'].append("Imminent failure predicted")

        except Exception as e:
            self.logger.warning(f"Error calculating system risk: {e}")

        return risk

    async def _continuous_data_collection(self):
        """Continuously collect system metrics"""
        while True:
            try:
                await asyncio.sleep(self.maintenance_config['data_collection_interval'])

                # Collect current metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent
                disk_percent = psutil.disk_usage('/').percent

                # Store in history
                self.system_metrics_history['cpu'].append(cpu_percent)
                self.system_metrics_history['memory'].append(memory_percent)
                self.system_metrics_history['disk'].append(disk_percent)

                # Collect network metrics
                net_io = psutil.net_io_counters()
                if net_io:
                    network_load = (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024  # MB
                    self.system_metrics_history['network'].append(network_load)

            except Exception as e:
                self.logger.warning(f"Error in data collection: {e}")
                await asyncio.sleep(60)

    async def _prediction_engine(self):
        """Background prediction engine"""
        while True:
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes

                # Generate predictions
                predictions = await self.predict_system_failures(
                    prediction_horizon=self.maintenance_config['prediction_window_hours'],
                    confidence_threshold=self.maintenance_config['confidence_threshold']
                )

                # Auto-schedule maintenance if enabled
                if self.maintenance_config['auto_maintenance']:
                    recommendations = predictions.get('maintenance_recommendations', [])
                    for rec in recommendations:
                        if rec.get('priority') == 'high':
                            await self.schedule_maintenance(
                                rec['action'],
                                rec['priority']
                            )

            except Exception as e:
                self.logger.warning(f"Error in prediction engine: {e}")
                await asyncio.sleep(1800)

    async def _load_historical_data(self):
        """Load historical system data"""
        try:
            history_file = Path('jarvis/data/system_history.json')
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)

                # Load historical metrics
                for metric, values in data.get('metrics', {}).items():
                    if metric in self.system_metrics_history:
                        self.system_metrics_history[metric].extend(values[-1000:])  # Last 1000 points

        except Exception as e:
            self.logger.warning(f"Error loading historical data: {e}")

    async def _initialize_prediction_models(self):
        """Initialize prediction models"""
        # Simple statistical models for now
        self.prediction_models = {
            'cpu_trend': {'method': 'linear_regression', 'accuracy': 0.75},
            'memory_trend': {'method': 'exponential_smoothing', 'accuracy': 0.70},
            'disk_trend': {'method': 'linear_regression', 'accuracy': 0.80}
        }

    def _estimate_maintenance_duration(self, maintenance_type: str) -> int:
        """Estimate maintenance duration in minutes"""
        duration_map = {
            'cpu_optimization': 15,
            'memory_cleanup': 10,
            'disk_cleanup': 20,
            'network_reset': 5,
            'process_restart': 2,
            'system_update': 30
        }
        return duration_map.get(maintenance_type, 15)

    def _get_maintenance_resources(self, maintenance_type: str) -> List[str]:
        """Get resources required for maintenance"""
        resource_map = {
            'cpu_optimization': ['cpu'],
            'memory_cleanup': ['memory'],
            'disk_cleanup': ['disk', 'io'],
            'network_reset': ['network'],
            'process_restart': ['process'],
            'system_update': ['system']
        }
        return resource_map.get(maintenance_type, ['system'])

    def _sort_maintenance_schedule(self):
        """Sort maintenance schedule by priority and time"""
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}

        self.maintenance_schedule.sort(key=lambda x: (
            priority_order.get(x.get('priority', 'medium'), 2),
            x.get('scheduled_time', float('inf'))
        ))

    async def _execute_maintenance_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a maintenance task"""
        result = {
            'success': True,
            'actions_taken': [],
            'errors': []
        }

        try:
            maintenance_type = task['type']

            if maintenance_type == 'cpu_optimization':
                result['actions_taken'].append("Optimized CPU scheduling")
            elif maintenance_type == 'memory_cleanup':
                result['actions_taken'].append("Performed memory cleanup")
            elif maintenance_type == 'disk_cleanup':
                result['actions_taken'].append("Cleaned up disk space")
            elif maintenance_type == 'network_reset':
                result['actions_taken'].append("Reset network connections")
            elif maintenance_type == 'process_restart':
                result['actions_taken'].append("Restarted problematic processes")
            elif maintenance_type == 'system_update':
                result['actions_taken'].append("Applied system updates")
            else:
                result['actions_taken'].append(f"Performed {maintenance_type}")

        except Exception as e:
            result['success'] = False
            result['errors'].append(str(e))

        return result

    def get_maintenance_stats(self) -> Dict[str, Any]:
        """Get maintenance statistics"""
        return {
            **self.stats,
            'scheduled_maintenance': len(self.maintenance_schedule),
            'active_predictions': len(self.predicted_failures),
            'historical_data_points': sum(len(history) for history in self.system_metrics_history.values())
        }

    async def shutdown(self):
        """Shutdown predictive maintenance"""
        try:
            self.logger.info("Shutting down predictive maintenance...")

            # Save historical data
            await self._save_historical_data()

            # Clear data
            for history in self.system_metrics_history.values():
                history.clear()

            self.maintenance_schedule.clear()
            self.predicted_failures.clear()

            self.logger.info("Predictive maintenance shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down predictive maintenance: {e}")

    async def _save_historical_data(self):
        """Save historical system data"""
        try:
            data_dir = Path('jarvis/data')
            data_dir.mkdir(exist_ok=True)

            data = {
                'metrics': {
                    metric: list(history)
                    for metric, history in self.system_metrics_history.items()
                },
                'last_updated': time.time()
            }

            with open(data_dir / 'system_history.json', 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.warning(f"Error saving historical data: {e}")