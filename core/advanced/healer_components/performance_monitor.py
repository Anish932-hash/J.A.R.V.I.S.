"""
J.A.R.V.I.S. Performance Monitor
Advanced real-time performance monitoring and alerting system
"""

import os
import time
import asyncio
import logging
import psutil
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import threading
import json
from pathlib import Path
from collections import deque


class PerformanceMonitor:
    """
    Ultra-advanced performance monitoring system with real-time metrics,
    anomaly detection, and intelligent alerting
    """

    def __init__(self, application_healer):
        self.healer = application_healer
        self.jarvis = application_healer.jarvis
        self.logger = logging.getLogger('JARVIS.PerformanceMonitor')

        # Monitoring configuration
        self.monitor_config = {
            'monitoring_enabled': True,
            'collection_interval': 10,  # seconds
            'alert_thresholds': {
                'cpu_percent': 80,
                'memory_percent': 85,
                'disk_percent': 90,
                'network_errors': 100
            },
            'anomaly_detection': True,
            'historical_analysis': True,
            'alert_cooldown': 300  # 5 minutes
        }

        # Monitoring data
        self.metrics_history = {
            'cpu': deque(maxlen=3600),  # 1 hour at 1 sample/second
            'memory': deque(maxlen=3600),
            'disk': deque(maxlen=3600),
            'network': deque(maxlen=3600),
            'processes': deque(maxlen=100)
        }

        self.active_alerts = {}
        self.monitoring_thread = None
        self.monitoring_active = False

        # Performance statistics
        self.stats = {
            'monitoring_sessions': 0,
            'alerts_generated': 0,
            'anomalies_detected': 0,
            'performance_trends': 0,
            'avg_response_time': 0.0,
            'uptime_percentage': 100.0
        }

    async def initialize(self):
        """Initialize performance monitor"""
        try:
            self.logger.info("Initializing performance monitor...")
            await self._setup_monitoring_infrastructure()
            self.start_monitoring()
            self.logger.info("Performance monitor initialized")
        except Exception as e:
            self.logger.error(f"Error initializing performance monitor: {e}")
            raise

    def start_monitoring(self):
        """Start performance monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.stats['monitoring_sessions'] += 1

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        try:
            metrics = {
                'timestamp': time.time(),
                'cpu': {
                    'percent': psutil.cpu_percent(interval=1),
                    'cores': psutil.cpu_count(),
                    'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None
                },
                'memory': {
                    'total_gb': psutil.virtual_memory().total / (1024**3),
                    'available_gb': psutil.virtual_memory().available / (1024**3),
                    'used_percent': psutil.virtual_memory().percent
                },
                'disk': {
                    'total_gb': psutil.disk_usage('/').total / (1024**3),
                    'free_gb': psutil.disk_usage('/').free / (1024**3),
                    'used_percent': psutil.disk_usage('/').percent
                },
                'network': {
                    'bytes_sent_mb': psutil.net_io_counters().bytes_sent / (1024 * 1024) if psutil.net_io_counters() else 0,
                    'bytes_recv_mb': psutil.net_io_counters().bytes_recv / (1024 * 1024) if psutil.net_io_counters() else 0
                },
                'processes': {
                    'total': len(psutil.pids()),
                    'running': len([p for p in psutil.process_iter(['status']) if p.info['status'] == 'running'])
                }
            }

            return metrics

        except Exception as e:
            self.logger.warning(f"Error getting current metrics: {e}")
            return {'error': str(e)}

    async def analyze_performance_trends(self, time_window: int = 3600) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        try:
            analysis = {
                'time_window': time_window,
                'trends': {},
                'anomalies': [],
                'predictions': {},
                'recommendations': []
            }

            # Analyze CPU trends
            if self.metrics_history['cpu']:
                cpu_values = list(self.metrics_history['cpu'])
                analysis['trends']['cpu'] = self._analyze_metric_trend(cpu_values, 'cpu')

            # Analyze memory trends
            if self.metrics_history['memory']:
                memory_values = list(self.metrics_history['memory'])
                analysis['trends']['memory'] = self._analyze_metric_trend(memory_values, 'memory')

            # Detect anomalies
            analysis['anomalies'] = await self._detect_anomalies()

            # Generate recommendations
            analysis['recommendations'] = await self._generate_performance_recommendations(analysis)

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing performance trends: {e}")
            return {'error': str(e)}

    async def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for performance alerts"""
        alerts = []

        try:
            current_metrics = await self.get_current_metrics()

            # CPU alerts
            cpu_percent = current_metrics['cpu']['percent']
            if cpu_percent > self.monitor_config['alert_thresholds']['cpu_percent']:
                alerts.append({
                    'type': 'cpu_high_usage',
                    'severity': 'high' if cpu_percent > 95 else 'medium',
                    'message': f'CPU usage at {cpu_percent:.1f}%',
                    'current_value': cpu_percent,
                    'threshold': self.monitor_config['alert_thresholds']['cpu_percent'],
                    'timestamp': time.time()
                })

            # Memory alerts
            memory_percent = current_metrics['memory']['used_percent']
            if memory_percent > self.monitor_config['alert_thresholds']['memory_percent']:
                alerts.append({
                    'type': 'memory_high_usage',
                    'severity': 'high' if memory_percent > 95 else 'medium',
                    'message': f'Memory usage at {memory_percent:.1f}%',
                    'current_value': memory_percent,
                    'threshold': self.monitor_config['alert_thresholds']['memory_percent'],
                    'timestamp': time.time()
                })

            # Disk alerts
            disk_percent = current_metrics['disk']['used_percent']
            if disk_percent > self.monitor_config['alert_thresholds']['disk_percent']:
                alerts.append({
                    'type': 'disk_high_usage',
                    'severity': 'high',
                    'message': f'Disk usage at {disk_percent:.1f}%',
                    'current_value': disk_percent,
                    'threshold': self.monitor_config['alert_thresholds']['disk_percent'],
                    'timestamp': time.time()
                })

            # Update active alerts
            for alert in alerts:
                alert_key = f"{alert['type']}_{int(time.time())}"
                self.active_alerts[alert_key] = alert
                self.stats['alerts_generated'] += 1

        except Exception as e:
            self.logger.warning(f"Error checking alerts: {e}")

        return alerts

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect metrics
                current_time = time.time()

                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.metrics_history['cpu'].append((current_time, cpu_percent))

                # Memory metrics
                memory_percent = psutil.virtual_memory().percent
                self.metrics_history['memory'].append((current_time, memory_percent))

                # Disk metrics
                disk_percent = psutil.disk_usage('/').percent
                self.metrics_history['disk'].append((current_time, disk_percent))

                # Network metrics
                if psutil.net_io_counters():
                    network_load = (psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv) / (1024 * 1024)
                    self.metrics_history['network'].append((current_time, network_load))

                # Process metrics
                process_count = len(psutil.pids())
                self.metrics_history['processes'].append((current_time, process_count))

                # Check for alerts (in thread-safe way)
                asyncio.run_coroutine_threadsafe(self.check_alerts(), asyncio.get_event_loop())

                # Sleep for collection interval
                time.sleep(self.monitor_config['collection_interval'])

            except Exception as e:
                self.logger.warning(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Wait before retry

    def _analyze_metric_trend(self, values: List[Tuple[float, float]], metric_name: str) -> Dict[str, Any]:
        """Analyze trend for a specific metric"""
        if len(values) < 2:
            return {'trend': 'insufficient_data'}

        # Extract values
        timestamps, metric_values = zip(*values)

        # Calculate trend
        start_value = metric_values[0]
        end_value = metric_values[-1]
        change = end_value - start_value
        change_percent = (change / start_value) * 100 if start_value != 0 else 0

        # Determine trend direction
        if abs(change_percent) < 5:
            trend = 'stable'
        elif change_percent > 5:
            trend = 'increasing'
        else:
            trend = 'decreasing'

        # Calculate volatility
        avg_value = sum(metric_values) / len(metric_values)
        variance = sum((v - avg_value) ** 2 for v in metric_values) / len(metric_values)
        volatility = variance ** 0.5

        return {
            'trend': trend,
            'change_percent': change_percent,
            'start_value': start_value,
            'end_value': end_value,
            'average_value': avg_value,
            'volatility': volatility,
            'min_value': min(metric_values),
            'max_value': max(metric_values)
        }

    async def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect performance anomalies"""
        anomalies = []

        try:
            # Simple anomaly detection based on statistical thresholds
            for metric_name, history in self.metrics_history.items():
                if len(history) < 10:
                    continue

                values = [v for t, v in history]
                avg_value = sum(values) / len(values)
                std_dev = (sum((v - avg_value) ** 2 for v in values) / len(values)) ** 0.5

                # Check recent values for anomalies
                recent_values = values[-5:]  # Last 5 values
                for i, value in enumerate(recent_values):
                    z_score = abs(value - avg_value) / std_dev if std_dev > 0 else 0
                    if z_score > 3:  # 3 standard deviations
                        anomalies.append({
                            'metric': metric_name,
                            'type': 'statistical_anomaly',
                            'severity': 'high' if z_score > 4 else 'medium',
                            'value': value,
                            'expected_range': (avg_value - 2*std_dev, avg_value + 2*std_dev),
                            'z_score': z_score,
                            'timestamp': history[-5 + i][0]
                        })

        except Exception as e:
            self.logger.warning(f"Error detecting anomalies: {e}")

        return anomalies

    async def _generate_performance_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate performance recommendations"""
        recommendations = []

        try:
            # CPU recommendations
            cpu_trend = analysis.get('trends', {}).get('cpu', {})
            if cpu_trend.get('trend') == 'increasing':
                recommendations.append({
                    'category': 'cpu',
                    'priority': 'medium',
                    'recommendation': 'Consider optimizing CPU-intensive operations',
                    'expected_impact': 'reduce_cpu_usage'
                })

            # Memory recommendations
            memory_trend = analysis.get('trends', {}).get('memory', {})
            if memory_trend.get('trend') == 'increasing':
                recommendations.append({
                    'category': 'memory',
                    'priority': 'high',
                    'recommendation': 'Implement memory optimization techniques',
                    'expected_impact': 'reduce_memory_usage'
                })

            # Anomaly-based recommendations
            anomalies = analysis.get('anomalies', [])
            if anomalies:
                recommendations.append({
                    'category': 'anomaly',
                    'priority': 'high',
                    'recommendation': f'Investigate {len(anomalies)} detected performance anomalies',
                    'expected_impact': 'improve_stability'
                })

        except Exception as e:
            self.logger.warning(f"Error generating recommendations: {e}")

        return recommendations

    async def _setup_monitoring_infrastructure(self):
        """Setup monitoring infrastructure"""
        Path('jarvis/monitoring').mkdir(parents=True, exist_ok=True)
        Path('jarvis/alerts').mkdir(parents=True, exist_ok=True)

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            **self.stats,
            'active_alerts': len(self.active_alerts),
            'monitoring_active': self.monitoring_active,
            'metrics_collected': sum(len(history) for history in self.metrics_history.values()),
            'alerts_today': len([a for a in self.active_alerts.values()
                               if time.time() - a['timestamp'] < 86400])
        }

    async def shutdown(self):
        """Shutdown performance monitor"""
        try:
            self.logger.info("Shutting down performance monitor...")
            self.stop_monitoring()

            # Save monitoring data
            await self._save_monitoring_data()

            self.logger.info("Performance monitor shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down performance monitor: {e}")

    async def _save_monitoring_data(self):
        """Save monitoring data"""
        try:
            data_file = Path('jarvis/data/monitoring_history.json')
            data_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert deques to lists for JSON serialization
            data = {
                'metrics_history': {
                    metric: list(history)
                    for metric, history in self.metrics_history.items()
                },
                'active_alerts': self.active_alerts,
                'stats': self.stats,
                'last_updated': time.time()
            }

            with open(data_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            self.logger.warning(f"Error saving monitoring data: {e}")