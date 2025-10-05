"""
J.A.R.V.I.S. Advanced Health Reporter
Comprehensive system health monitoring, reporting, and analytics
"""

import os
import time
import json
import psutil
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor


@dataclass
class HealthMetric:
    """Represents a health metric"""
    name: str
    category: str
    current_value: float
    unit: str
    status: str  # healthy, warning, critical
    threshold_warning: float
    threshold_critical: float
    trend: str  # improving, stable, degrading
    history: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, value)


@dataclass
class HealthAlert:
    """Represents a health alert"""
    alert_id: str
    severity: str
    category: str
    message: str
    timestamp: float
    resolved: bool = False
    resolved_at: Optional[float] = None
    recommendations: List[str] = field(default_factory=list)


class HealthAnalyzer:
    """Analyzes system health metrics"""

    def __init__(self):
        self.baseline_metrics = {}
        self.metric_history = {}
        self.alerts = []

    def analyze_overall_health(self, metrics: Dict[str, HealthMetric]) -> Dict[str, Any]:
        """Analyze overall system health"""
        analysis = {
            "overall_score": 100.0,
            "status": "healthy",
            "critical_issues": 0,
            "warning_issues": 0,
            "trending_metrics": [],
            "risk_assessment": "low"
        }

        critical_count = 0
        warning_count = 0
        total_score = 0
        metric_count = 0

        for metric in metrics.values():
            metric_count += 1

            # Calculate metric score
            if metric.status == "critical":
                critical_count += 1
                score_penalty = 30
            elif metric.status == "warning":
                warning_count += 1
                score_penalty = 10
            else:
                score_penalty = 0

            # Trend penalty
            if metric.trend == "degrading":
                score_penalty += 5

            total_score += max(0, 10 - score_penalty)

            # Track trending metrics
            if metric.trend != "stable":
                analysis["trending_metrics"].append({
                    "name": metric.name,
                    "trend": metric.trend,
                    "change_percent": self._calculate_trend_change(metric)
                })

        if metric_count > 0:
            analysis["overall_score"] = (total_score / metric_count) * 10

        analysis["critical_issues"] = critical_count
        analysis["warning_issues"] = warning_count

        # Determine overall status
        if critical_count > 0:
            analysis["status"] = "critical"
            analysis["risk_assessment"] = "high"
        elif warning_count > 2:
            analysis["status"] = "warning"
            analysis["risk_assessment"] = "medium"
        elif warning_count > 0:
            analysis["status"] = "warning"
            analysis["risk_assessment"] = "low"
        else:
            analysis["status"] = "healthy"
            analysis["risk_assessment"] = "minimal"

        return analysis

    def _calculate_trend_change(self, metric: HealthMetric) -> float:
        """Calculate trend change percentage"""
        if len(metric.history) < 2:
            return 0.0

        recent_values = [v for _, v in metric.history[-10:]]  # Last 10 readings
        if len(recent_values) < 2:
            return 0.0

        older_avg = statistics.mean(recent_values[:len(recent_values)//2])
        newer_avg = statistics.mean(recent_values[len(recent_values)//2:])

        if older_avg == 0:
            return 0.0

        return ((newer_avg - older_avg) / older_avg) * 100

    def generate_recommendations(self, analysis: Dict[str, Any],
                               metrics: Dict[str, HealthMetric]) -> List[str]:
        """Generate health recommendations"""
        recommendations = []

        # Overall health recommendations
        if analysis["status"] == "critical":
            recommendations.append("Immediate attention required - critical system issues detected")
            recommendations.append("Consider system restart or professional maintenance")
        elif analysis["status"] == "warning":
            recommendations.append("Monitor system closely - warning conditions detected")
            recommendations.append("Schedule maintenance to address potential issues")

        # Specific metric recommendations
        for metric in metrics.values():
            if metric.status == "critical":
                if "cpu" in metric.name.lower():
                    recommendations.append("High CPU usage detected - optimize running processes")
                elif "memory" in metric.name.lower():
                    recommendations.append("High memory usage - close unnecessary applications")
                elif "disk" in metric.name.lower():
                    recommendations.append("Low disk space - clean up storage or add capacity")

            elif metric.status == "warning":
                if metric.trend == "degrading":
                    recommendations.append(f"{metric.name} is trending downward - investigate cause")

        # Trend-based recommendations
        for trend_info in analysis["trending_metrics"]:
            if trend_info["trend"] == "degrading":
                recommendations.append(f"Address degrading trend in {trend_info['name']}")

        # General recommendations
        if analysis["overall_score"] < 70:
            recommendations.append("Overall system health is poor - comprehensive maintenance recommended")
        elif analysis["overall_score"] < 85:
            recommendations.append("System health could be improved with regular maintenance")

        return recommendations

    def detect_anomalies(self, metrics: Dict[str, HealthMetric]) -> List[Dict[str, Any]]:
        """Detect anomalous metric behavior"""
        anomalies = []

        for metric in metrics.values():
            # Check for sudden spikes
            if len(metric.history) >= 5:
                recent_values = [v for _, v in metric.history[-5:]]
                avg_recent = statistics.mean(recent_values)
                std_recent = statistics.stdev(recent_values) if len(recent_values) > 1 else 0

                current_value = metric.current_value

                # Check if current value is outlier
                if std_recent > 0:
                    z_score = abs(current_value - avg_recent) / std_recent
                    if z_score > 3:  # 3-sigma rule
                        anomalies.append({
                            "metric": metric.name,
                            "type": "sudden_spike" if current_value > avg_recent else "sudden_drop",
                            "deviation": z_score,
                            "current_value": current_value,
                            "expected_range": f"{avg_recent-2*std_recent:.2f} - {avg_recent+2*std_recent:.2f}"
                        })

        return anomalies


class ReportGenerator:
    """Generates comprehensive health reports"""

    def __init__(self):
        self.report_templates = {
            "summary": self._generate_summary_report,
            "detailed": self._generate_detailed_report,
            "trends": self._generate_trends_report,
            "alerts": self._generate_alerts_report
        }

    def generate_report(self, report_type: str, health_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a health report"""
        if report_type not in self.report_templates:
            return {"error": f"Unknown report type: {report_type}"}

        return self.report_templates[report_type](health_data)

    def _generate_summary_report(self, health_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary health report"""
        analysis = health_data.get("analysis", {})
        metrics = health_data.get("metrics", {})

        return {
            "report_type": "summary",
            "generated_at": time.time(),
            "overall_health": analysis.get("status", "unknown"),
            "health_score": analysis.get("overall_score", 0),
            "critical_issues": analysis.get("critical_issues", 0),
            "warning_issues": analysis.get("warning_issues", 0),
            "total_metrics": len(metrics),
            "recommendations_count": len(health_data.get("recommendations", [])),
            "risk_assessment": analysis.get("risk_assessment", "unknown")
        }

    def _generate_detailed_report(self, health_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed health report"""
        analysis = health_data.get("analysis", {})
        metrics = health_data.get("metrics", {})
        recommendations = health_data.get("recommendations", [])

        # Group metrics by status
        metrics_by_status = {
            "healthy": [],
            "warning": [],
            "critical": []
        }

        for metric in metrics.values():
            metrics_by_status[metric.status].append({
                "name": metric.name,
                "category": metric.category,
                "current_value": metric.current_value,
                "unit": metric.unit,
                "trend": metric.trend
            })

        return {
            "report_type": "detailed",
            "generated_at": time.time(),
            "overall_analysis": analysis,
            "metrics_by_status": metrics_by_status,
            "trending_metrics": analysis.get("trending_metrics", []),
            "recommendations": recommendations,
            "anomalies": health_data.get("anomalies", []),
            "system_info": health_data.get("system_info", {})
        }

    def _generate_trends_report(self, health_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trends analysis report"""
        metrics = health_data.get("metrics", {})

        trends_analysis = {
            "improving": [],
            "stable": [],
            "degrading": []
        }

        for metric in metrics.values():
            trend_data = {
                "name": metric.name,
                "current_value": metric.current_value,
                "history_points": len(metric.history),
                "change_percent": self._calculate_metric_change(metric)
            }

            trends_analysis[metric.trend].append(trend_data)

        return {
            "report_type": "trends",
            "generated_at": time.time(),
            "trends_analysis": trends_analysis,
            "trend_summary": {
                "improving_count": len(trends_analysis["improving"]),
                "stable_count": len(trends_analysis["stable"]),
                "degrading_count": len(trends_analysis["degrading"])
            }
        }

    def _generate_alerts_report(self, health_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate alerts-focused report"""
        alerts = health_data.get("alerts", [])

        # Group alerts by severity
        alerts_by_severity = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": []
        }

        for alert in alerts:
            severity = alert.get("severity", "low")
            alerts_by_severity[severity].append(alert)

        return {
            "report_type": "alerts",
            "generated_at": time.time(),
            "total_alerts": len(alerts),
            "alerts_by_severity": alerts_by_severity,
            "active_alerts": [a for a in alerts if not a.get("resolved", False)],
            "resolved_alerts": [a for a in alerts if a.get("resolved", True)]
        }

    def _calculate_metric_change(self, metric: HealthMetric) -> float:
        """Calculate percentage change in metric over time"""
        if len(metric.history) < 2:
            return 0.0

        oldest_value = metric.history[0][1]
        newest_value = metric.history[-1][1]

        if oldest_value == 0:
            return 0.0

        return ((newest_value - oldest_value) / oldest_value) * 100


class HealthReporter:
    """Advanced system health reporting and analytics"""

    def __init__(self, healer):
        self.healer = healer
        self.jarvis = healer.jarvis
        self.logger = logging.getLogger('JARVIS.HealthReporter')

        # Health monitoring components
        self.health_analyzer = HealthAnalyzer()
        self.report_generator = ReportGenerator()

        # Health data
        self.health_metrics: Dict[str, HealthMetric] = {}
        self.health_alerts: List[HealthAlert] = []
        self.health_history: List[Dict[str, Any]] = []

        # Configuration
        self.monitoring_interval = 60  # 1 minute
        self.history_retention_days = 30
        self.alert_thresholds = {
            "cpu_percent": {"warning": 70, "critical": 90},
            "memory_percent": {"warning": 80, "critical": 95},
            "disk_percent": {"warning": 85, "critical": 95}
        }

    async def initialize(self):
        """Initialize the advanced health reporter"""
        try:
            self.logger.info("Initializing advanced health reporter...")

            # Initialize health metrics
            await self._initialize_health_metrics()

            # Load historical data
            await self._load_health_history()

            # Start monitoring
            asyncio.create_task(self._health_monitoring_loop())

            self.logger.info("Advanced health reporter initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing health reporter: {e}")
            raise

    async def _initialize_health_metrics(self):
        """Initialize health metrics tracking"""
        # CPU metrics
        self.health_metrics["cpu_usage"] = HealthMetric(
            name="CPU Usage",
            category="performance",
            current_value=0.0,
            unit="%",
            status="healthy",
            threshold_warning=self.alert_thresholds["cpu_percent"]["warning"],
            threshold_critical=self.alert_thresholds["cpu_percent"]["critical"],
            trend="stable"
        )

        # Memory metrics
        self.health_metrics["memory_usage"] = HealthMetric(
            name="Memory Usage",
            category="performance",
            current_value=0.0,
            unit="%",
            status="healthy",
            threshold_warning=self.alert_thresholds["memory_percent"]["warning"],
            threshold_critical=self.alert_thresholds["memory_percent"]["critical"],
            trend="stable"
        )

        # Disk metrics
        self.health_metrics["disk_usage"] = HealthMetric(
            name="Disk Usage",
            category="storage",
            current_value=0.0,
            unit="%",
            status="healthy",
            threshold_warning=self.alert_thresholds["disk_percent"]["warning"],
            threshold_critical=self.alert_thresholds["disk_percent"]["critical"],
            trend="stable"
        )

        # Network metrics
        self.health_metrics["network_connections"] = HealthMetric(
            name="Network Connections",
            category="network",
            current_value=0.0,
            unit="count",
            status="healthy",
            threshold_warning=100,
            threshold_critical=500,
            trend="stable"
        )

    async def generate_report(self, report_type: str = "summary") -> Dict[str, Any]:
        """Generate comprehensive health report"""
        try:
            # Update current metrics
            await self._update_health_metrics()

            # Analyze health
            analysis = self.health_analyzer.analyze_overall_health(self.health_metrics)

            # Generate recommendations
            recommendations = self.health_analyzer.generate_recommendations(
                analysis, self.health_metrics
            )

            # Detect anomalies
            anomalies = self.health_analyzer.detect_anomalies(self.health_metrics)

            # Get system information
            system_info = await self._get_system_info()

            # Prepare health data
            health_data = {
                "analysis": analysis,
                "metrics": self.health_metrics,
                "recommendations": recommendations,
                "anomalies": anomalies,
                "alerts": [alert.__dict__ for alert in self.health_alerts[-10:]],  # Last 10 alerts
                "system_info": system_info,
                "generated_at": time.time()
            }

            # Generate report
            report = self.report_generator.generate_report(report_type, health_data)

            # Store in history
            self.health_history.append({
                "timestamp": time.time(),
                "report_type": report_type,
                "overall_health": analysis["status"],
                "health_score": analysis["overall_score"],
                "critical_issues": analysis["critical_issues"],
                "warning_issues": analysis["warning_issues"]
            })

            # Cleanup old history
            cutoff_time = time.time() - (self.history_retention_days * 24 * 3600)
            self.health_history = [
                h for h in self.health_history if h["timestamp"] > cutoff_time
            ]

            return report

        except Exception as e:
            self.logger.error(f"Error generating health report: {e}")
            return {
                "error": str(e),
                "report_type": report_type,
                "generated_at": time.time()
            }

    async def _update_health_metrics(self):
        """Update current health metrics"""
        try:
            # CPU usage
            cpu_metric = self.health_metrics["cpu_usage"]
            cpu_metric.current_value = psutil.cpu_percent(interval=1)
            cpu_metric.history.append((time.time(), cpu_metric.current_value))
            cpu_metric.status = self._determine_metric_status(cpu_metric)

            # Memory usage
            memory_metric = self.health_metrics["memory_usage"]
            memory_metric.current_value = psutil.virtual_memory().percent
            memory_metric.history.append((time.time(), memory_metric.current_value))
            memory_metric.status = self._determine_metric_status(memory_metric)

            # Disk usage
            disk_metric = self.health_metrics["disk_usage"]
            disk_metric.current_value = psutil.disk_usage('/').percent
            disk_metric.history.append((time.time(), disk_metric.current_value))
            disk_metric.status = self._determine_metric_status(disk_metric)

            # Network connections
            network_metric = self.health_metrics["network_connections"]
            network_metric.current_value = len(psutil.net_connections())
            network_metric.history.append((time.time(), network_metric.current_value))
            network_metric.status = self._determine_metric_status(network_metric)

            # Update trends
            for metric in self.health_metrics.values():
                metric.trend = self._calculate_metric_trend(metric)

                # Keep history manageable
                if len(metric.history) > 1000:
                    metric.history = metric.history[-1000:]

        except Exception as e:
            self.logger.error(f"Error updating health metrics: {e}")

    def _determine_metric_status(self, metric: HealthMetric) -> str:
        """Determine status of a health metric"""
        if metric.current_value >= metric.threshold_critical:
            return "critical"
        elif metric.current_value >= metric.threshold_warning:
            return "warning"
        else:
            return "healthy"

    def _calculate_metric_trend(self, metric: HealthMetric) -> str:
        """Calculate trend for a metric"""
        if len(metric.history) < 5:
            return "stable"

        recent_values = [v for _, v in metric.history[-10:]]
        if len(recent_values) < 5:
            return "stable"

        # Simple trend calculation
        first_half = recent_values[:len(recent_values)//2]
        second_half = recent_values[len(recent_values)//2:]

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        diff = second_avg - first_avg
        threshold = metric.current_value * 0.05  # 5% change threshold

        if diff > threshold:
            return "degrading"
        elif diff < -threshold:
            return "improving"
        else:
            return "stable"

    async def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        try:
            return {
                "platform": os.uname().sysname if hasattr(os, 'uname') else 'unknown',
                "hostname": os.uname().nodename if hasattr(os, 'uname') else 'unknown',
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "disk_total": psutil.disk_usage('/').total,
                "boot_time": psutil.boot_time(),
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                "process_count": len(psutil.pids())
            }
        except Exception as e:
            return {"error": str(e)}

    async def _health_monitoring_loop(self):
        """Continuous health monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.monitoring_interval)

                # Update metrics
                await self._update_health_metrics()

                # Check for alerts
                await self._check_health_alerts()

                # Auto-generate reports periodically
                if int(time.time()) % 3600 == 0:  # Every hour
                    report = await self.generate_report("summary")
                    if report.get("overall_health") in ["critical", "warning"]:
                        self.logger.warning(f"Health alert: System health is {report['overall_health']}")

            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retry

    async def _check_health_alerts(self):
        """Check for health alerts and create them"""
        for metric in self.health_metrics.values():
            if metric.status in ["warning", "critical"]:
                # Check if we already have an active alert for this metric
                existing_alert = next(
                    (a for a in self.health_alerts
                     if not a.resolved and a.category == metric.category and metric.name in a.message),
                    None
                )

                if not existing_alert:
                    # Create new alert
                    alert = HealthAlert(
                        alert_id=f"alert_{int(time.time())}_{hash(metric.name) % 1000}",
                        severity=metric.status,
                        category=metric.category,
                        message=f"{metric.name} is {metric.status}: {metric.current_value}{metric.unit}",
                        timestamp=time.time(),
                        recommendations=self._generate_alert_recommendations(metric)
                    )

                    self.health_alerts.append(alert)

                    # Log alert
                    self.logger.warning(f"Health alert created: {alert.message}")

            else:
                # Check if we need to resolve existing alerts
                for alert in self.health_alerts:
                    if (not alert.resolved and
                        alert.category == metric.category and
                        metric.name in alert.message):

                        alert.resolved = True
                        alert.resolved_at = time.time()
                        self.logger.info(f"Health alert resolved: {alert.message}")

    def _generate_alert_recommendations(self, metric: HealthMetric) -> List[str]:
        """Generate recommendations for a health alert"""
        recommendations = []

        if "cpu" in metric.name.lower():
            recommendations.extend([
                "Close unnecessary applications",
                "Check for runaway processes",
                "Consider upgrading CPU"
            ])
        elif "memory" in metric.name.lower():
            recommendations.extend([
                "Close memory-intensive applications",
                "Clear system cache",
                "Consider adding more RAM"
            ])
        elif "disk" in metric.name.lower():
            recommendations.extend([
                "Delete unnecessary files",
                "Empty recycle bin/trash",
                "Move files to external storage"
            ])
        elif "network" in metric.name.lower():
            recommendations.extend([
                "Check network connections",
                "Close network-intensive applications",
                "Restart network services"
            ])

        return recommendations

    def get_health_stats(self) -> Dict[str, Any]:
        """Get health reporting statistics"""
        total_reports = len(self.health_history)
        critical_reports = sum(1 for h in self.health_history if h.get("overall_health") == "critical")
        warning_reports = sum(1 for h in self.health_history if h.get("overall_health") == "warning")

        return {
            "total_reports": total_reports,
            "critical_reports": critical_reports,
            "warning_reports": warning_reports,
            "healthy_reports": total_reports - critical_reports - warning_reports,
            "active_alerts": len([a for a in self.health_alerts if not a.resolved]),
            "resolved_alerts": len([a for a in self.health_alerts if a.resolved]),
            "monitored_metrics": len(self.health_metrics)
        }

    async def _load_health_history(self):
        """Load historical health data"""
        try:
            history_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'health_history.json')
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.health_history = data.get("history", [])

                    # Load alerts
                    alerts_data = data.get("alerts", [])
                    for alert_dict in alerts_data:
                        alert = HealthAlert(**alert_dict)
                        self.health_alerts.append(alert)

        except Exception as e:
            self.logger.debug(f"Could not load health history: {e}")

    async def generate_health_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive health report

        Returns:
            Dict containing health report data
        """
        try:
            # Update metrics first
            await self._update_health_metrics()

            report = {
                "timestamp": time.time(),
                "summary": {
                    "overall_status": "healthy",
                    "issues_count": 0,
                    "warnings_count": 0
                },
                "issues": [],
                "warnings": [],
                "metrics": {},
                "recommendations": []
            }

            # Analyze metrics
            for metric_name, metric in self.health_metrics.items():
                if metric.status == "critical":
                    report["summary"]["issues_count"] += 1
                    report["summary"]["overall_status"] = "critical"
                    report["issues"].append({
                        "metric": metric_name,
                        "message": f"Critical: {metric.name} is at {metric.current_value}{metric.unit}",
                        "value": metric.current_value,
                        "threshold": metric.threshold_critical
                    })
                elif metric.status == "warning":
                    if report["summary"]["overall_status"] == "healthy":
                        report["summary"]["overall_status"] = "warning"
                    report["summary"]["warnings_count"] += 1
                    report["warnings"].append({
                        "metric": metric_name,
                        "message": f"Warning: {metric.name} is at {metric.current_value}{metric.unit}",
                        "value": metric.current_value,
                        "threshold": metric.threshold_warning
                    })

                # Add metric data
                report["metrics"][metric_name] = {
                    "name": metric.name,
                    "value": metric.current_value,
                    "unit": metric.unit,
                    "status": metric.status,
                    "trend": metric.trend
                }

            # Generate recommendations based on issues
            if report["issues"]:
                report["recommendations"].append("Address critical health issues immediately")
            if report["warnings"]:
                report["recommendations"].append("Review warning conditions to prevent issues")
            if not report["issues"] and not report["warnings"]:
                report["recommendations"].append("System health is optimal")

            return report

        except Exception as e:
            self.logger.error(f"Error generating health report: {e}")
            return {
                "timestamp": time.time(),
                "summary": {
                    "overall_status": "unknown",
                    "issues_count": 1,
                    "warnings_count": 0
                },
                "issues": [{"error": str(e), "severity": "critical"}],
                "warnings": [],
                "metrics": {},
                "recommendations": ["Investigate health reporting system"]
            }

    async def shutdown(self):
        """Shutdown the advanced health reporter"""
        try:
            self.logger.info("Shutting down advanced health reporter...")

            # Save health data
            await self._save_health_data()

            self.logger.info("Advanced health reporter shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down health reporter: {e}")

    async def _save_health_data(self):
        """Save health data"""
        try:
            data_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'health_history.json')

            os.makedirs(os.path.dirname(data_file), exist_ok=True)

            data = {
                "history": self.health_history[-1000:],  # Last 1000 reports
                "alerts": [alert.__dict__ for alert in self.health_alerts[-100:]],  # Last 100 alerts
                "last_updated": time.time()
            }

            with open(data_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Error saving health data: {e}")