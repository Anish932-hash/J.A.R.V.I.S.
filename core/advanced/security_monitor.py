"""
J.A.R.V.I.S. Advanced Security Monitor
Real-time security monitoring, threat detection, and automated response system
"""

import sys
import os
import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Set
import logging
import json
import hashlib
import psutil
from datetime import datetime, timedelta
import re

try:
    import scapy.all as scapy
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

try:
    import yara
    YARA_AVAILABLE = True
except ImportError:
    YARA_AVAILABLE = False


class ThreatIntelligence:
    """Threat intelligence gathering and analysis"""

    def __init__(self):
        self.threat_feeds = {}
        self.indicators = set()
        self.logger = logging.getLogger('JARVIS.ThreatIntelligence')

    async def update_threat_feeds(self) -> Dict[str, Any]:
        """Update threat intelligence feeds"""
        try:
            # In a real implementation, this would fetch from various threat feeds
            # For now, simulate with mock data

            mock_indicators = {
                'malicious_ips': ['192.168.1.100', '10.0.0.50'],
                'suspicious_domains': ['malicious-site.com', 'phishing-domain.net'],
                'known_malware_hashes': [
                    'a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3',
                    'b615a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae4'
                ],
                'suspicious_processes': ['suspicious.exe', 'malware.bin']
            }

            self.indicators.update(mock_indicators['malicious_ips'])
            self.indicators.update(mock_indicators['suspicious_domains'])
            self.indicators.update(mock_indicators['known_malware_hashes'])
            self.indicators.update(mock_indicators['suspicious_processes'])

            return {
                'feeds_updated': len(self.threat_feeds),
                'indicators_loaded': len(self.indicators),
                'last_update': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error updating threat feeds: {e}")
            return {}

    def check_indicator(self, value: str) -> bool:
        """Check if a value matches known threat indicators"""
        return value in self.indicators


class BehavioralAnalyzer:
    """Behavioral analysis for anomaly detection"""

    def __init__(self):
        self.baseline_metrics = {}
        self.anomaly_threshold = 3.0  # Standard deviations
        self.logger = logging.getLogger('JARVIS.BehavioralAnalyzer')

    def establish_baseline(self, metrics_history: Dict[str, List[float]]) -> bool:
        """Establish behavioral baseline"""
        try:
            for metric, values in metrics_history.items():
                if len(values) >= 10:  # Need minimum data points
                    mean = sum(values) / len(values)
                    variance = sum((x - mean) ** 2 for x in values) / len(values)
                    std_dev = variance ** 0.5

                    self.baseline_metrics[metric] = {
                        'mean': mean,
                        'std_dev': std_dev,
                        'min_normal': mean - (self.anomaly_threshold * std_dev),
                        'max_normal': mean + (self.anomaly_threshold * std_dev)
                    }

            self.logger.info(f"Baseline established for {len(self.baseline_metrics)} metrics")
            return True

        except Exception as e:
            self.logger.error(f"Error establishing baseline: {e}")
            return False

    def detect_anomalies(self, current_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect behavioral anomalies"""
        try:
            anomalies = []

            for metric, value in current_metrics.items():
                if metric in self.baseline_metrics:
                    baseline = self.baseline_metrics[metric]

                    if value < baseline['min_normal'] or value > baseline['max_normal']:
                        # Calculate deviation
                        deviation = abs(value - baseline['mean']) / baseline['std_dev'] if baseline['std_dev'] > 0 else 0

                        anomalies.append({
                            'metric': metric,
                            'value': value,
                            'expected_range': [baseline['min_normal'], baseline['max_normal']],
                            'deviation': deviation,
                            'severity': 'high' if deviation > 5 else 'medium' if deviation > 3 else 'low',
                            'timestamp': datetime.now().isoformat()
                        })

            return anomalies

        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            return []


class NetworkMonitor:
    """Network traffic monitoring and analysis"""

    def __init__(self):
        self.captured_packets = []
        self.suspicious_connections = []
        self.logger = logging.getLogger('JARVIS.NetworkMonitor')

    async def start_monitoring(self) -> bool:
        """Start network monitoring"""
        try:
            if not SCAPY_AVAILABLE:
                self.logger.warning("Scapy not available - network monitoring limited")
                return False

            # In a real implementation, this would start packet capture
            # For now, simulate monitoring
            self.logger.info("Network monitoring started")
            return True

        except Exception as e:
            self.logger.error(f"Error starting network monitoring: {e}")
            return False

    def analyze_connections(self) -> List[Dict[str, Any]]:
        """Analyze network connections"""
        try:
            connections = psutil.net_connections()
            suspicious = []

            for conn in connections:
                if conn.status == 'ESTABLISHED':
                    # Check for suspicious patterns
                    if self._is_suspicious_connection(conn):
                        suspicious.append({
                            'local_address': f"{conn.laddr.ip}:{conn.laddr.port}",
                            'remote_address': f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "unknown",
                            'status': conn.status,
                            'pid': conn.pid,
                            'risk_level': 'high' if self._is_high_risk_port(conn.raddr.port if conn.raddr else 0) else 'medium'
                        })

            return suspicious

        except Exception as e:
            self.logger.error(f"Error analyzing connections: {e}")
            return []

    def _is_suspicious_connection(self, conn) -> bool:
        """Check if connection is suspicious"""
        try:
            if not conn.raddr:
                return False

            remote_ip = conn.raddr.ip
            remote_port = conn.raddr.port

            # Check for private IP ranges connecting to suspicious ports
            if remote_ip.startswith(('192.168.', '10.', '172.')) and remote_port in [22, 23, 3389]:
                return True

            # Check for connections to known suspicious ports
            suspicious_ports = [6667, 6697, 7000, 8000, 8080]  # IRC, common malware ports
            if remote_port in suspicious_ports:
                return True

            return False

        except Exception:
            return False

    def _is_high_risk_port(self, port: int) -> bool:
        """Check if port is high risk"""
        high_risk_ports = [22, 23, 3389, 5900, 6667]  # SSH, Telnet, RDP, VNC, IRC
        return port in high_risk_ports


class FileIntegrityMonitor:
    """File integrity monitoring and change detection"""

    def __init__(self):
        self.baseline_hashes = {}
        self.monitored_paths = []
        self.logger = logging.getLogger('JARVIS.FileIntegrityMonitor')

    def establish_baseline(self, paths: List[str]) -> bool:
        """Establish file integrity baseline"""
        try:
            for path in paths:
                if os.path.exists(path):
                    if os.path.isfile(path):
                        self.baseline_hashes[path] = self._calculate_file_hash(path)
                    elif os.path.isdir(path):
                        self._hash_directory(path)

            self.monitored_paths = paths
            self.logger.info(f"Baseline established for {len(self.baseline_hashes)} files")
            return True

        except Exception as e:
            self.logger.error(f"Error establishing baseline: {e}")
            return False

    def check_integrity(self) -> List[Dict[str, Any]]:
        """Check file integrity"""
        try:
            violations = []

            for path in self.monitored_paths:
                if os.path.exists(path):
                    if os.path.isfile(path):
                        current_hash = self._calculate_file_hash(path)
                        if path in self.baseline_hashes and current_hash != self.baseline_hashes[path]:
                            violations.append({
                                'path': path,
                                'type': 'modified',
                                'severity': 'high',
                                'timestamp': datetime.now().isoformat()
                            })
                    elif os.path.isdir(path):
                        violations.extend(self._check_directory_integrity(path))

            return violations

        except Exception as e:
            self.logger.error(f"Error checking integrity: {e}")
            return []

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        try:
            sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()

        except Exception:
            return ""

    def _hash_directory(self, dir_path: str):
        """Hash all files in directory"""
        try:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    self.baseline_hashes[file_path] = self._calculate_file_hash(file_path)

        except Exception as e:
            self.logger.error(f"Error hashing directory {dir_path}: {e}")

    def _check_directory_integrity(self, dir_path: str) -> List[Dict[str, Any]]:
        """Check directory integrity"""
        try:
            violations = []

            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file_path not in self.baseline_hashes:
                        violations.append({
                            'path': file_path,
                            'type': 'new_file',
                            'severity': 'medium',
                            'timestamp': datetime.now().isoformat()
                        })
                    else:
                        current_hash = self._calculate_file_hash(file_path)
                        if current_hash != self.baseline_hashes[file_path]:
                            violations.append({
                                'path': file_path,
                                'type': 'modified',
                                'severity': 'high',
                                'timestamp': datetime.now().isoformat()
                            })

            return violations

        except Exception as e:
            self.logger.error(f"Error checking directory integrity: {e}")
            return []


class AdvancedSecurityMonitor:
    """Advanced security monitoring and threat detection system"""

    def __init__(self, development_engine):
        self.development_engine = development_engine
        self.jarvis = development_engine.jarvis if hasattr(development_engine, 'jarvis') else None
        self.logger = logging.getLogger('JARVIS.AdvancedSecurityMonitor')

        # Security components
        self.threat_intelligence = ThreatIntelligence()
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.network_monitor = NetworkMonitor()
        self.file_integrity_monitor = FileIntegrityMonitor()

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.alerts = []
        self.baseline_established = False

        # Configuration
        self.monitoring_interval = 30  # seconds
        self.alert_threshold = 5  # Maximum alerts per hour

    async def initialize(self):
        """Initialize advanced security monitor"""
        try:
            self.logger.info("Initializing Advanced Security Monitor...")

            # Update threat intelligence
            await self.threat_intelligence.update_threat_feeds()

            # Start network monitoring
            await self.network_monitor.start_monitoring()

            # Establish file integrity baseline
            critical_paths = [
                os.path.join(os.path.dirname(__file__), '..', '..', 'jarvis'),
                os.path.join(os.path.dirname(__file__), '..', '..', 'config'),
                os.path.join(os.path.dirname(__file__), '..', '..', 'data')
            ]
            self.file_integrity_monitor.establish_baseline(critical_paths)

            self.logger.info("Advanced Security Monitor initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing security monitor: {e}")
            return False

    def start_monitoring(self) -> str:
        """Start security monitoring"""
        try:
            if self.is_monitoring:
                return "Security monitoring already active"

            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()

            self.logger.info("Security monitoring started")
            return "Security monitoring started"

        except Exception as e:
            self.logger.error(f"Error starting monitoring: {e}")
            return f"Error starting monitoring: {e}"

    def stop_monitoring(self) -> str:
        """Stop security monitoring"""
        try:
            if not self.is_monitoring:
                return "Security monitoring not active"

            self.is_monitoring = False
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)

            self.logger.info("Security monitoring stopped")
            return "Security monitoring stopped"

        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {e}")
            return f"Error stopping monitoring: {e}"

    def _monitoring_loop(self):
        """Main security monitoring loop"""
        try:
            while self.is_monitoring:
                # Perform security checks
                self._perform_security_checks()

                # Clean up old alerts
                self._cleanup_old_alerts()

                # Sleep before next check
                time.sleep(self.monitoring_interval)

        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {e}")
            self.is_monitoring = False

    def _perform_security_checks(self):
        """Perform comprehensive security checks"""
        try:
            current_time = datetime.now()

            # 1. Behavioral analysis
            if self.baseline_established and self.jarvis and hasattr(self.jarvis, 'system_monitor'):
                current_metrics = {
                    'cpu_percent': self.jarvis.system_monitor.current_readings.get('cpu', {}).get('percent', 0),
                    'memory_percent': self.jarvis.system_monitor.current_readings.get('memory', {}).get('percent', 0),
                    'disk_percent': self.jarvis.system_monitor.current_readings.get('disk', {}).get('main_percent', 0)
                }

                anomalies = self.behavioral_analyzer.detect_anomalies(current_metrics)
                for anomaly in anomalies:
                    self._create_alert('behavioral_anomaly', f"Anomaly detected in {anomaly['metric']}", anomaly)

            # 2. Network monitoring
            suspicious_connections = self.network_monitor.analyze_connections()
            for conn in suspicious_connections:
                self._create_alert('suspicious_connection',
                                 f"Suspicious network connection: {conn['remote_address']}",
                                 conn)

            # 3. File integrity check
            integrity_violations = self.file_integrity_monitor.check_integrity()
            for violation in integrity_violations:
                severity = 'critical' if violation['severity'] == 'high' else 'warning'
                self._create_alert('file_integrity_violation',
                                 f"File integrity violation: {violation['path']} ({violation['type']})",
                                 violation, severity)

            # 4. Process monitoring
            suspicious_processes = self._check_processes()
            for proc in suspicious_processes:
                self._create_alert('suspicious_process',
                                 f"Suspicious process detected: {proc['name']} (PID: {proc['pid']})",
                                 proc)

            # 5. System vulnerability check
            vulnerabilities = self._check_vulnerabilities()
            for vuln in vulnerabilities:
                self._create_alert('system_vulnerability',
                                 f"System vulnerability: {vuln['description']}",
                                 vuln, 'high')

        except Exception as e:
            self.logger.error(f"Error performing security checks: {e}")

    def _create_alert(self, alert_type: str, message: str, details: Dict[str, Any] = None, severity: str = 'medium'):
        """Create a security alert"""
        try:
            alert = {
                'id': f"{alert_type}_{int(time.time())}_{len(self.alerts)}",
                'type': alert_type,
                'message': message,
                'severity': severity,
                'timestamp': datetime.now().isoformat(),
                'details': details or {},
                'status': 'active'
            }

            self.alerts.append(alert)

            # Log alert
            self.logger.warning(f"Security Alert [{severity.upper()}]: {message}")

            # Trigger automated response if critical
            if severity in ['high', 'critical']:
                asyncio.run(self._trigger_automated_response(alert))

        except Exception as e:
            self.logger.error(f"Error creating alert: {e}")

    async def _trigger_automated_response(self, alert: Dict[str, Any]):
        """Trigger automated response to security alert"""
        try:
            alert_type = alert['type']

            if alert_type == 'suspicious_connection':
                # Block suspicious IP (simulation)
                self.logger.info(f"Automated response: Blocking suspicious connection {alert['details'].get('remote_address')}")

            elif alert_type == 'file_integrity_violation':
                # Create backup or quarantine file (simulation)
                self.logger.info(f"Automated response: File integrity violation handled for {alert['details'].get('path')}")

            elif alert_type == 'suspicious_process':
                # Terminate suspicious process (simulation)
                self.logger.info(f"Automated response: Suspicious process {alert['details'].get('name')} monitored")

            # In a real system, this would trigger actual security responses

        except Exception as e:
            self.logger.error(f"Error triggering automated response: {e}")

    def _check_processes(self) -> List[Dict[str, Any]]:
        """Check for suspicious processes"""
        try:
            suspicious = []

            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    # Check for known suspicious process names
                    if any(suspicious_name in proc.info['name'].lower() for suspicious_name in
                           ['trojan', 'virus', 'malware', 'keylogger', 'ransomware']):
                        suspicious.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cpu_percent': proc.info['cpu_percent'],
                            'memory_percent': proc.info['memory_percent'],
                            'risk_level': 'high'
                        })

                    # Check for processes with unusually high resource usage
                    elif proc.info['cpu_percent'] > 90 or proc.info['memory_percent'] > 80:
                        suspicious.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cpu_percent': proc.info['cpu_percent'],
                            'memory_percent': proc.info['memory_percent'],
                            'risk_level': 'medium'
                        })

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return suspicious

        except Exception as e:
            self.logger.error(f"Error checking processes: {e}")
            return []

    def _check_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Check for system vulnerabilities"""
        try:
            vulnerabilities = []

            # Check if system is up to date (simplified check)
            try:
                import platform
                system = platform.system().lower()

                if system == 'windows':
                    # Check Windows update status (simulation)
                    vulnerabilities.append({
                        'type': 'outdated_system',
                        'description': 'System updates may be pending',
                        'severity': 'medium',
                        'recommendation': 'Run Windows Update'
                    })

            except Exception:
                pass

            # Check for weak permissions on critical files
            critical_files = [
                os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'jarvis.json'),
                os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'master.key')
            ]

            for file_path in critical_files:
                if os.path.exists(file_path):
                    try:
                        # Check if file is world-readable (simplified check)
                        import stat
                        file_stat = os.stat(file_path)
                        if bool(file_stat.st_mode & stat.S_IRGRP) or bool(file_stat.st_mode & stat.S_IROTH):
                            vulnerabilities.append({
                                'type': 'weak_permissions',
                                'description': f'Weak permissions on {os.path.basename(file_path)}',
                                'severity': 'high',
                                'recommendation': 'Restrict file permissions'
                            })
                    except Exception:
                        pass

            return vulnerabilities

        except Exception as e:
            self.logger.error(f"Error checking vulnerabilities: {e}")
            return []

    def _cleanup_old_alerts(self):
        """Clean up old alerts"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.alerts = [
                alert for alert in self.alerts
                if datetime.fromisoformat(alert['timestamp']) > cutoff_time
            ]

        except Exception as e:
            self.logger.error(f"Error cleaning up old alerts: {e}")

    def establish_behavioral_baseline(self) -> bool:
        """Establish behavioral baseline for anomaly detection"""
        try:
            if not self.jarvis or not hasattr(self.jarvis, 'system_monitor'):
                return False

            # Collect historical data (would need actual historical data in production)
            mock_history = {
                'cpu_percent': [45.2, 42.1, 48.3, 43.7, 46.8, 44.5, 47.2, 41.9, 49.1, 45.6],
                'memory_percent': [62.3, 64.1, 61.8, 63.2, 65.4, 62.9, 64.7, 63.1, 66.2, 64.8],
                'disk_percent': [45.2, 45.3, 45.1, 45.4, 45.2, 45.3, 45.1, 45.4, 45.2, 45.3]
            }

            success = self.behavioral_analyzer.establish_baseline(mock_history)
            if success:
                self.baseline_established = True

            return success

        except Exception as e:
            self.logger.error(f"Error establishing behavioral baseline: {e}")
            return False

    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        try:
            # Count alerts by severity
            alert_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
            for alert in self.alerts:
                severity = alert.get('severity', 'medium')
                alert_counts[severity] += 1

            # Calculate overall security score
            total_alerts = sum(alert_counts.values())
            security_score = max(0, 100 - (total_alerts * 5) - (alert_counts['high'] * 10) - (alert_counts['critical'] * 20))

            # Determine overall status
            if security_score >= 90:
                status = 'secure'
            elif security_score >= 70:
                status = 'good'
            elif security_score >= 50:
                status = 'warning'
            else:
                status = 'critical'

            return {
                'overall_status': status,
                'security_score': security_score,
                'alert_counts': alert_counts,
                'total_alerts': total_alerts,
                'monitoring_active': self.is_monitoring,
                'baseline_established': self.baseline_established,
                'threat_indicators': len(self.threat_intelligence.indicators),
                'last_check': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting security status: {e}")
            return {}

    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent security alerts"""
        try:
            return sorted(self.alerts, key=lambda x: x['timestamp'], reverse=True)[:limit]

        except Exception as e:
            self.logger.error(f"Error getting recent alerts: {e}")
            return []

    def run_security_scan(self) -> Dict[str, Any]:
        """Run comprehensive security scan"""
        try:
            scan_results = {
                'scan_start': datetime.now().isoformat(),
                'checks_performed': [],
                'issues_found': [],
                'recommendations': []
            }

            # 1. File integrity check
            integrity_issues = self.file_integrity_monitor.check_integrity()
            scan_results['checks_performed'].append('file_integrity')
            scan_results['issues_found'].extend(integrity_issues)

            # 2. Process analysis
            process_issues = self._check_processes()
            scan_results['checks_performed'].append('process_analysis')
            scan_results['issues_found'].extend(process_issues)

            # 3. Network analysis
            network_issues = self.network_monitor.analyze_connections()
            scan_results['checks_performed'].append('network_analysis')
            scan_results['issues_found'].extend(network_issues)

            # 4. Vulnerability assessment
            vulnerabilities = self._check_vulnerabilities()
            scan_results['checks_performed'].append('vulnerability_assessment')
            scan_results['issues_found'].extend(vulnerabilities)

            # Generate recommendations
            scan_results['recommendations'] = self._generate_security_recommendations(scan_results['issues_found'])

            scan_results['scan_end'] = datetime.now().isoformat()
            scan_results['total_issues'] = len(scan_results['issues_found'])

            return scan_results

        except Exception as e:
            self.logger.error(f"Error running security scan: {e}")
            return {}

    def _generate_security_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations based on findings"""
        try:
            recommendations = []

            issue_types = set(issue.get('type', 'unknown') for issue in issues)

            if 'modified' in issue_types:
                recommendations.append("Review and verify file integrity changes")
                recommendations.append("Consider enabling file change auditing")

            if 'suspicious_process' in issue_types:
                recommendations.append("Investigate suspicious processes and terminate if malicious")
                recommendations.append("Implement process whitelisting")

            if 'suspicious_connection' in issue_types:
                recommendations.append("Review network connections and block suspicious IPs")
                recommendations.append("Implement network traffic monitoring")

            if 'system_vulnerability' in issue_types:
                recommendations.append("Apply system updates and security patches")
                recommendations.append("Review and strengthen system configurations")

            if not issues:
                recommendations.append("Security scan completed successfully - no issues found")
                recommendations.append("Continue regular security monitoring")

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating security recommendations: {e}")
            return []

    async def shutdown(self):
        """Shutdown advanced security monitor"""
        try:
            self.stop_monitoring()
            self.logger.info("Advanced Security Monitor shutdown")
        except Exception as e:
            self.logger.error(f"Error shutting down security monitor: {e}")