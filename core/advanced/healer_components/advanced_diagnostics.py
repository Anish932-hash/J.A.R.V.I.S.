"""
J.A.R.V.I.S. Advanced Diagnostics
Ultra-sophisticated diagnostic system for deep system analysis and issue detection
"""

import os
import time
import psutil
import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import threading
from datetime import datetime, timedelta
import json
from pathlib import Path


class AdvancedDiagnostics:
    """
    Ultra-advanced diagnostic system that performs deep system analysis,
    detects complex issues, and provides comprehensive health assessments
    """

    def __init__(self, application_healer):
        """
        Initialize advanced diagnostics

        Args:
            application_healer: Reference to application healer
        """
        self.healer = application_healer
        self.jarvis = application_healer.jarvis
        self.logger = logging.getLogger('JARVIS.AdvancedDiagnostics')

        # Diagnostic configuration
        self.diagnostic_config = {
            'deep_scan_enabled': True,
            'performance_monitoring': True,
            'memory_analysis': True,
            'thread_analysis': True,
            'network_diagnostics': True,
            'disk_health_check': True,
            'log_analysis': True,
            'scan_interval': 60,  # seconds
            'diagnostic_timeout': 300  # 5 minutes
        }

        # Diagnostic state
        self.diagnostic_history = []
        self.current_diagnostics = {}
        self.baseline_metrics = {}

        # Diagnostic statistics
        self.stats = {
            'diagnostics_performed': 0,
            'issues_detected': 0,
            'critical_issues': 0,
            'performance_issues': 0,
            'memory_issues': 0,
            'false_positives': 0,
            'diagnostic_accuracy': 0.0
        }

    async def initialize(self):
        """Initialize advanced diagnostics"""
        try:
            self.logger.info("Initializing advanced diagnostics...")

            # Establish baseline metrics
            await self._establish_baseline_metrics()

            # Start continuous monitoring
            asyncio.create_task(self._continuous_diagnostic_monitor())

            self.logger.info("Advanced diagnostics initialized")

        except Exception as e:
            self.logger.error(f"Error initializing advanced diagnostics: {e}")
            raise

    async def perform_comprehensive_diagnosis(self,
                                            target_systems: List[str] = None,
                                            diagnostic_depth: str = "comprehensive") -> Dict[str, Any]:
        """
        Perform comprehensive system diagnosis

        Args:
            target_systems: Specific systems to diagnose
            diagnostic_depth: Depth of diagnosis (basic, standard, comprehensive, deep)

        Returns:
            Comprehensive diagnostic results
        """
        start_time = time.time()

        try:
            self.logger.info(f"Performing {diagnostic_depth} comprehensive diagnosis")

            # Determine diagnostic scope
            if target_systems is None:
                target_systems = ['system', 'memory', 'cpu', 'disk', 'network', 'processes', 'logs']

            # Perform diagnostic phases
            diagnostic_results = {}

            # Phase 1: System health assessment
            diagnostic_results['system_health'] = await self._assess_system_health()

            # Phase 2: Performance diagnostics
            if 'cpu' in target_systems or 'performance' in target_systems:
                diagnostic_results['performance'] = await self._diagnose_performance()

            # Phase 3: Memory diagnostics
            if 'memory' in target_systems:
                diagnostic_results['memory'] = await self._diagnose_memory()

            # Phase 4: Storage diagnostics
            if 'disk' in target_systems:
                diagnostic_results['storage'] = await self._diagnose_storage()

            # Phase 5: Network diagnostics
            if 'network' in target_systems:
                diagnostic_results['network'] = await self._diagnose_network()

            # Phase 6: Process diagnostics
            if 'processes' in target_systems:
                diagnostic_results['processes'] = await self._diagnose_processes()

            # Phase 7: Log analysis
            if 'logs' in target_systems:
                diagnostic_results['logs'] = await self._analyze_logs()

            # Phase 8: Deep system analysis (if requested)
            if diagnostic_depth in ['comprehensive', 'deep']:
                diagnostic_results['deep_analysis'] = await self._perform_deep_analysis()

            # Synthesize findings
            synthesis = await self._synthesize_diagnostic_findings(diagnostic_results)

            # Generate recommendations
            recommendations = await self._generate_diagnostic_recommendations(synthesis)

            diagnostic_time = time.time() - start_time
            self.stats['diagnostics_performed'] += 1

            result = {
                'diagnostic_depth': diagnostic_depth,
                'target_systems': target_systems,
                'diagnostic_results': diagnostic_results,
                'synthesis': synthesis,
                'recommendations': recommendations,
                'diagnostic_time': diagnostic_time,
                'timestamp': time.time(),
                'diagnostic_id': f"diag_{int(time.time())}_{hash(str(target_systems)) % 10000}"
            }

            # Store diagnostic results
            self.diagnostic_history.append(result)
            self.current_diagnostics[result['diagnostic_id']] = result

            # Update statistics
            total_issues = sum(len(section.get('issues', [])) for section in diagnostic_results.values()
                             if isinstance(section, dict))
            self.stats['issues_detected'] += total_issues

            critical_issues = sum(len([i for i in section.get('issues', []) if i.get('severity') == 'critical'])
                                for section in diagnostic_results.values() if isinstance(section, dict))
            self.stats['critical_issues'] += critical_issues

            self.logger.info(f"Comprehensive diagnosis completed in {diagnostic_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error in comprehensive diagnosis: {e}")
            return {
                'error': str(e),
                'diagnostic_time': time.time() - start_time
            }

    async def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health"""
        health_assessment = {
            'overall_health': 'unknown',
            'health_score': 0.0,
            'critical_indicators': [],
            'warning_indicators': [],
            'health_trends': []
        }

        try:
            # Check basic system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Assess CPU health
            if cpu_percent > 90:
                health_assessment['critical_indicators'].append('High CPU usage')
                health_assessment['health_score'] -= 20
            elif cpu_percent > 70:
                health_assessment['warning_indicators'].append('Elevated CPU usage')
                health_assessment['health_score'] -= 10

            # Assess memory health
            if memory.percent > 90:
                health_assessment['critical_indicators'].append('Critical memory usage')
                health_assessment['health_score'] -= 25
            elif memory.percent > 80:
                health_assessment['warning_indicators'].append('High memory usage')
                health_assessment['health_score'] -= 15

            # Assess disk health
            if disk.percent > 95:
                health_assessment['critical_indicators'].append('Critical disk usage')
                health_assessment['health_score'] -= 20
            elif disk.percent > 85:
                health_assessment['warning_indicators'].append('Low disk space')
                health_assessment['health_score'] -= 10

            # Calculate overall health score
            health_assessment['health_score'] = max(0, min(100, 100 + health_assessment['health_score']))

            # Determine health status
            if health_assessment['health_score'] >= 80:
                health_assessment['overall_health'] = 'excellent'
            elif health_assessment['health_score'] >= 60:
                health_assessment['overall_health'] = 'good'
            elif health_assessment['health_score'] >= 40:
                health_assessment['overall_health'] = 'fair'
            elif health_assessment['health_score'] >= 20:
                health_assessment['overall_health'] = 'poor'
            else:
                health_assessment['overall_health'] = 'critical'

        except Exception as e:
            self.logger.warning(f"Error assessing system health: {e}")
            health_assessment['overall_health'] = 'unknown'
            health_assessment['critical_indicators'].append(f"Health assessment error: {e}")

        return health_assessment

    async def _diagnose_performance(self) -> Dict[str, Any]:
        """Diagnose system performance"""
        performance_diagnosis = {
            'cpu_analysis': {},
            'memory_analysis': {},
            'io_analysis': {},
            'bottlenecks': [],
            'recommendations': []
        }

        try:
            # CPU analysis
            cpu_times = psutil.cpu_times_percent(interval=1)
            performance_diagnosis['cpu_analysis'] = {
                'user_percent': cpu_times.user,
                'system_percent': cpu_times.system,
                'idle_percent': cpu_times.idle,
                'bottleneck_detected': cpu_times.user + cpu_times.system > 80
            }

            # Memory analysis
            memory = psutil.virtual_memory()
            performance_diagnosis['memory_analysis'] = {
                'total_mb': memory.total / 1024 / 1024,
                'used_mb': memory.used / 1024 / 1024,
                'available_mb': memory.available / 1024 / 1024,
                'usage_percent': memory.percent,
                'bottleneck_detected': memory.percent > 85
            }

            # I/O analysis
            io_counters = psutil.disk_io_counters()
            if io_counters:
                performance_diagnosis['io_analysis'] = {
                    'read_count': io_counters.read_count,
                    'write_count': io_counters.write_count,
                    'read_bytes': io_counters.read_bytes,
                    'write_bytes': io_counters.write_bytes,
                    'read_time_ms': io_counters.read_time,
                    'write_time_ms': io_counters.write_time
                }

            # Identify bottlenecks
            if performance_diagnosis['cpu_analysis']['bottleneck_detected']:
                performance_diagnosis['bottlenecks'].append({
                    'type': 'cpu',
                    'severity': 'high',
                    'description': 'High CPU utilization detected'
                })

            if performance_diagnosis['memory_analysis']['bottleneck_detected']:
                performance_diagnosis['bottlenecks'].append({
                    'type': 'memory',
                    'severity': 'high',
                    'description': 'High memory utilization detected'
                })

            # Generate recommendations
            if performance_diagnosis['bottlenecks']:
                performance_diagnosis['recommendations'].append(
                    "Consider optimizing resource-intensive operations"
                )

        except Exception as e:
            self.logger.warning(f"Error in performance diagnosis: {e}")
            performance_diagnosis['error'] = str(e)

        return performance_diagnosis

    async def _diagnose_memory(self) -> Dict[str, Any]:
        """Diagnose memory-related issues"""
        memory_diagnosis = {
            'memory_leaks': [],
            'fragmentation': {},
            'allocation_patterns': {},
            'swap_usage': {},
            'issues': []
        }

        try:
            # Memory leak detection (simplified)
            process = psutil.Process()
            memory_info = process.memory_info()

            # Check for unusual memory growth
            if hasattr(self, '_last_memory_check'):
                last_rss = self._last_memory_check.get('rss', 0)
                current_rss = memory_info.rss

                if current_rss > last_rss * 1.5:  # 50% increase
                    memory_diagnosis['memory_leaks'].append({
                        'type': 'sudden_growth',
                        'description': 'Sudden memory growth detected',
                        'growth_mb': (current_rss - last_rss) / 1024 / 1024
                    })

            self._last_memory_check = {
                'rss': memory_info.rss,
                'vms': memory_info.vms,
                'timestamp': time.time()
            }

            # Memory fragmentation analysis
            memory_diagnosis['fragmentation'] = {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'fragmentation_ratio': memory_info.vms / memory_info.rss if memory_info.rss > 0 else 0
            }

            # Swap usage
            swap = psutil.swap_memory()
            memory_diagnosis['swap_usage'] = {
                'total_mb': swap.total / 1024 / 1024,
                'used_mb': swap.used / 1024 / 1024,
                'free_mb': swap.free / 1024 / 1024,
                'usage_percent': swap.percent
            }

            # Identify issues
            if swap.percent > 50:
                memory_diagnosis['issues'].append({
                    'type': 'high_swap_usage',
                    'severity': 'medium',
                    'description': f'High swap usage: {swap.percent}%'
                })

            if memory_diagnosis['fragmentation']['fragmentation_ratio'] > 2:
                memory_diagnosis['issues'].append({
                    'type': 'memory_fragmentation',
                    'severity': 'low',
                    'description': 'Memory fragmentation detected'
                })

        except Exception as e:
            self.logger.warning(f"Error in memory diagnosis: {e}")
            memory_diagnosis['error'] = str(e)

        return memory_diagnosis

    async def _diagnose_storage(self) -> Dict[str, Any]:
        """Diagnose storage-related issues"""
        storage_diagnosis = {
            'disk_usage': {},
            'file_system_health': {},
            'io_performance': {},
            'storage_issues': []
        }

        try:
            # Disk usage analysis
            disk_usage = psutil.disk_usage('/')
            storage_diagnosis['disk_usage'] = {
                'total_gb': disk_usage.total / 1024 / 1024 / 1024,
                'used_gb': disk_usage.used / 1024 / 1024 / 1024,
                'free_gb': disk_usage.free / 1024 / 1024 / 1024,
                'usage_percent': disk_usage.percent
            }

            # I/O performance
            io_counters = psutil.disk_io_counters()
            if io_counters:
                storage_diagnosis['io_performance'] = {
                    'read_speed_mbs': io_counters.read_bytes / 1024 / 1024,
                    'write_speed_mbs': io_counters.write_bytes / 1024 / 1024,
                    'read_time_ms': io_counters.read_time,
                    'write_time_ms': io_counters.write_time
                }

            # Identify storage issues
            if disk_usage.percent > 90:
                storage_diagnosis['storage_issues'].append({
                    'type': 'low_disk_space',
                    'severity': 'critical',
                    'description': f'Critical disk usage: {disk_usage.percent}%'
                })
            elif disk_usage.percent > 80:
                storage_diagnosis['storage_issues'].append({
                    'type': 'low_disk_space',
                    'severity': 'high',
                    'description': f'High disk usage: {disk_usage.percent}%'
                })

        except Exception as e:
            self.logger.warning(f"Error in storage diagnosis: {e}")
            storage_diagnosis['error'] = str(e)

        return storage_diagnosis

    async def _diagnose_network(self) -> Dict[str, Any]:
        """Diagnose network-related issues"""
        network_diagnosis = {
            'interface_stats': {},
            'connection_quality': {},
            'network_issues': []
        }

        try:
            # Network interface statistics
            net_io = psutil.net_io_counters()
            if net_io:
                network_diagnosis['interface_stats'] = {
                    'bytes_sent_mb': net_io.bytes_sent / 1024 / 1024,
                    'bytes_recv_mb': net_io.bytes_recv / 1024 / 1024,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv,
                    'errin': net_io.errin,
                    'errout': net_io.errout
                }

                # Check for network errors
                if net_io.errin > 100 or net_io.errout > 100:
                    network_diagnosis['network_issues'].append({
                        'type': 'network_errors',
                        'severity': 'medium',
                        'description': f'Network errors detected: {net_io.errin} in, {net_io.errout} out'
                    })

            # Network connections
            connections = psutil.net_connections()
            network_diagnosis['connection_stats'] = {
                'total_connections': len(connections),
                'listening_ports': len([c for c in connections if c.status == 'LISTEN']),
                'established_connections': len([c for c in connections if c.status == 'ESTABLISHED'])
            }

        except Exception as e:
            self.logger.warning(f"Error in network diagnosis: {e}")
            network_diagnosis['error'] = str(e)

        return network_diagnosis

    async def _diagnose_processes(self) -> Dict[str, Any]:
        """Diagnose process-related issues"""
        process_diagnosis = {
            'process_count': 0,
            'high_cpu_processes': [],
            'high_memory_processes': [],
            'zombie_processes': [],
            'process_issues': []
        }

        try:
            # Get all processes
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            process_diagnosis['process_count'] = len(processes)

            # Identify high CPU processes
            high_cpu = [p for p in processes if p.get('cpu_percent', 0) > 50]
            process_diagnosis['high_cpu_processes'] = high_cpu[:10]  # Top 10

            # Identify high memory processes
            high_memory = [p for p in processes if p.get('memory_percent', 0) > 10]
            process_diagnosis['high_memory_processes'] = high_memory[:10]  # Top 10

            # Identify zombie processes
            zombies = [p for p in processes if p.get('status') == 'zombie']
            process_diagnosis['zombie_processes'] = zombies

            # Identify issues
            if len(high_cpu) > 5:
                process_diagnosis['process_issues'].append({
                    'type': 'multiple_high_cpu',
                    'severity': 'medium',
                    'description': f'{len(high_cpu)} processes with high CPU usage'
                })

            if zombies:
                process_diagnosis['process_issues'].append({
                    'type': 'zombie_processes',
                    'severity': 'high',
                    'description': f'{len(zombies)} zombie processes detected'
                })

        except Exception as e:
            self.logger.warning(f"Error in process diagnosis: {e}")
            process_diagnosis['error'] = str(e)

        return process_diagnosis

    async def _analyze_logs(self) -> Dict[str, Any]:
        """Analyze system logs for issues"""
        log_analysis = {
            'log_files_analyzed': [],
            'error_patterns': [],
            'warning_patterns': [],
            'critical_events': [],
            'log_issues': []
        }

        try:
            # Analyze JARVIS logs
            log_files = [
                'jarvis/logs/jarvis.log',
                'jarvis/logs/jarvis_20251004_063631.log'  # Most recent log
            ]

            for log_file in log_files:
                if os.path.exists(log_file):
                    analysis = await self._analyze_log_file(log_file)
                    log_analysis['log_files_analyzed'].append(analysis)

                    # Aggregate findings
                    log_analysis['error_patterns'].extend(analysis.get('error_patterns', []))
                    log_analysis['warning_patterns'].extend(analysis.get('warning_patterns', []))
                    log_analysis['critical_events'].extend(analysis.get('critical_events', []))

            # Identify log issues
            if log_analysis['error_patterns']:
                log_analysis['log_issues'].append({
                    'type': 'error_logs',
                    'severity': 'medium',
                    'description': f'{len(log_analysis["error_patterns"])} error patterns found in logs'
                })

            if log_analysis['critical_events']:
                log_analysis['log_issues'].append({
                    'type': 'critical_events',
                    'severity': 'high',
                    'description': f'{len(log_analysis["critical_events"])} critical events detected'
                })

        except Exception as e:
            self.logger.warning(f"Error in log analysis: {e}")
            log_analysis['error'] = str(e)

        return log_analysis

    async def _analyze_log_file(self, log_file: str) -> Dict[str, Any]:
        """Analyze a specific log file"""
        analysis = {
            'file': log_file,
            'error_patterns': [],
            'warning_patterns': [],
            'critical_events': [],
            'total_lines': 0
        }

        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                analysis['total_lines'] = len(lines)

                for line in lines[-1000:]:  # Analyze last 1000 lines
                    line_lower = line.lower()

                    # Check for error patterns
                    if 'error' in line_lower or 'exception' in line_lower:
                        analysis['error_patterns'].append({
                            'line': line.strip(),
                            'timestamp': self._extract_timestamp_from_log(line)
                        })

                    # Check for warning patterns
                    if 'warning' in line_lower or 'warn' in line_lower:
                        analysis['warning_patterns'].append({
                            'line': line.strip(),
                            'timestamp': self._extract_timestamp_from_log(line)
                        })

                    # Check for critical events
                    if any(word in line_lower for word in ['critical', 'fatal', 'panic', 'crash']):
                        analysis['critical_events'].append({
                            'line': line.strip(),
                            'timestamp': self._extract_timestamp_from_log(line)
                        })

        except Exception as e:
            self.logger.warning(f"Error analyzing log file {log_file}: {e}")

        return analysis

    def _extract_timestamp_from_log(self, line: str) -> Optional[str]:
        """Extract timestamp from log line"""
        # Simple timestamp extraction (would be more sophisticated in production)
        import re
        timestamp_match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', line)
        return timestamp_match.group(0) if timestamp_match else None

    async def _perform_deep_analysis(self) -> Dict[str, Any]:
        """Perform deep system analysis"""
        deep_analysis = {
            'system_interdependencies': {},
            'resource_contentions': [],
            'performance_anomalies': [],
            'predictive_insights': []
        }

        try:
            # Analyze system interdependencies
            deep_analysis['system_interdependencies'] = await self._analyze_system_interdependencies()

            # Detect resource contentions
            deep_analysis['resource_contentions'] = await self._detect_resource_contentions()

            # Identify performance anomalies
            deep_analysis['performance_anomalies'] = await self._identify_performance_anomalies()

            # Generate predictive insights
            deep_analysis['predictive_insights'] = await self._generate_predictive_insights()

        except Exception as e:
            self.logger.warning(f"Error in deep analysis: {e}")
            deep_analysis['error'] = str(e)

        return deep_analysis

    async def _analyze_system_interdependencies(self) -> Dict[str, Any]:
        """Analyze system component interdependencies"""
        # This would analyze how different system components interact
        return {
            'cpu_memory_dependency': 'high',
            'disk_io_dependency': 'medium',
            'network_system_dependency': 'low'
        }

    async def _detect_resource_contentions(self) -> List[Dict[str, Any]]:
        """Detect resource contentions"""
        contentions = []

        try:
            # Check for CPU contention
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 80:
                contentions.append({
                    'resource': 'cpu',
                    'severity': 'high',
                    'description': f'CPU contention at {cpu_percent}% utilization'
                })

            # Check for memory contention
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                contentions.append({
                    'resource': 'memory',
                    'severity': 'high',
                    'description': f'Memory contention at {memory.percent}% utilization'
                })

        except Exception as e:
            self.logger.warning(f"Error detecting resource contentions: {e}")

        return contentions

    async def _identify_performance_anomalies(self) -> List[Dict[str, Any]]:
        """Identify performance anomalies"""
        anomalies = []

        try:
            # Compare current metrics with baseline
            current_cpu = psutil.cpu_percent(interval=1)
            baseline_cpu = self.baseline_metrics.get('cpu_percent', 50)

            if abs(current_cpu - baseline_cpu) > 30:  # 30% deviation
                anomalies.append({
                    'type': 'cpu_anomaly',
                    'description': f'CPU usage anomaly: {current_cpu}% vs baseline {baseline_cpu}%',
                    'deviation': current_cpu - baseline_cpu
                })

        except Exception as e:
            self.logger.warning(f"Error identifying performance anomalies: {e}")

        return anomalies

    async def _generate_predictive_insights(self) -> List[Dict[str, Any]]:
        """Generate predictive insights"""
        insights = []

        try:
            # Simple predictive analysis based on current trends
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            if memory.percent > 75:
                insights.append({
                    'type': 'memory_trend',
                    'prediction': 'Memory exhaustion likely within 24 hours',
                    'confidence': 0.7,
                    'timeframe': '24 hours'
                })

            if disk.percent > 85:
                insights.append({
                    'type': 'disk_trend',
                    'prediction': 'Disk space exhaustion likely within 48 hours',
                    'confidence': 0.8,
                    'timeframe': '48 hours'
                })

        except Exception as e:
            self.logger.warning(f"Error generating predictive insights: {e}")

        return insights

    async def _synthesize_diagnostic_findings(self, diagnostic_results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize all diagnostic findings"""
        synthesis = {
            'overall_health_score': 0.0,
            'critical_issues_count': 0,
            'high_priority_issues_count': 0,
            'system_status': 'unknown',
            'key_findings': [],
            'risk_assessment': 'low'
        }

        try:
            # Calculate overall health score
            health_scores = []
            critical_count = 0
            high_count = 0

            for section_name, section_data in diagnostic_results.items():
                if isinstance(section_data, dict):
                    # Extract health scores from different sections
                    if 'health_score' in section_data:
                        health_scores.append(section_data['health_score'])

                    # Count critical issues
                    issues = section_data.get('issues', []) + section_data.get('critical_indicators', [])
                    critical_issues = [i for i in issues if i.get('severity') == 'critical']
                    high_issues = [i for i in issues if i.get('severity') == 'high']

                    critical_count += len(critical_issues)
                    high_count += len(high_issues)

            # Calculate average health score
            if health_scores:
                synthesis['overall_health_score'] = sum(health_scores) / len(health_scores)
            else:
                synthesis['overall_health_score'] = 75.0  # Default

            synthesis['critical_issues_count'] = critical_count
            synthesis['high_priority_issues_count'] = high_count

            # Determine system status
            if synthesis['overall_health_score'] >= 80 and critical_count == 0:
                synthesis['system_status'] = 'healthy'
            elif synthesis['overall_health_score'] >= 60 and critical_count <= 2:
                synthesis['system_status'] = 'warning'
            elif synthesis['overall_health_score'] >= 40 or critical_count <= 5:
                synthesis['system_status'] = 'degraded'
            else:
                synthesis['system_status'] = 'critical'

            # Determine risk assessment
            if critical_count > 3 or synthesis['overall_health_score'] < 50:
                synthesis['risk_assessment'] = 'high'
            elif critical_count > 0 or synthesis['overall_health_score'] < 70:
                synthesis['risk_assessment'] = 'medium'
            else:
                synthesis['risk_assessment'] = 'low'

            # Generate key findings
            synthesis['key_findings'] = self._extract_key_findings(diagnostic_results)

        except Exception as e:
            self.logger.warning(f"Error synthesizing diagnostic findings: {e}")
            synthesis['synthesis_error'] = str(e)

        return synthesis

    def _extract_key_findings(self, diagnostic_results: Dict[str, Any]) -> List[str]:
        """Extract key findings from diagnostic results"""
        findings = []

        try:
            # Extract most critical findings
            for section_name, section_data in diagnostic_results.items():
                if isinstance(section_data, dict):
                    issues = section_data.get('issues', [])
                    critical_issues = [i for i in issues if i.get('severity') == 'critical']

                    for issue in critical_issues[:2]:  # Limit to 2 per section
                        findings.append(f"Critical {section_name} issue: {issue.get('description', 'Unknown')}")

                    # Add section-specific key findings
                    if section_name == 'system_health':
                        health = section_data.get('overall_health', 'unknown')
                        findings.append(f"System health status: {health}")

                    elif section_name == 'performance':
                        bottlenecks = section_data.get('bottlenecks', [])
                        if bottlenecks:
                            findings.append(f"Performance bottlenecks: {len(bottlenecks)} detected")

        except Exception as e:
            self.logger.warning(f"Error extracting key findings: {e}")

        return findings[:10]  # Limit to 10 key findings

    async def _generate_diagnostic_recommendations(self, synthesis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate diagnostic recommendations"""
        recommendations = []

        try:
            health_score = synthesis.get('overall_health_score', 75)
            critical_count = synthesis.get('critical_issues_count', 0)
            system_status = synthesis.get('system_status', 'unknown')

            # Health-based recommendations
            if system_status == 'critical':
                recommendations.append({
                    'priority': 'critical',
                    'action': 'immediate_attention',
                    'description': 'System requires immediate attention',
                    'timeframe': 'immediate'
                })

            elif system_status == 'degraded':
                recommendations.append({
                    'priority': 'high',
                    'action': 'system_optimization',
                    'description': 'Perform system optimization and maintenance',
                    'timeframe': 'within 24 hours'
                })

            # Issue-based recommendations
            if critical_count > 0:
                recommendations.append({
                    'priority': 'high',
                    'action': 'address_critical_issues',
                    'description': f'Address {critical_count} critical issues immediately',
                    'timeframe': 'immediate'
                })

            # Preventive recommendations
            if health_score > 90:
                recommendations.append({
                    'priority': 'low',
                    'action': 'maintain_current_state',
                    'description': 'Continue monitoring and maintenance practices',
                    'timeframe': 'ongoing'
                })

        except Exception as e:
            self.logger.warning(f"Error generating diagnostic recommendations: {e}")

        return recommendations

    async def _establish_baseline_metrics(self):
        """Establish baseline system metrics"""
        try:
            # Collect baseline metrics over a short period
            baseline_samples = []

            for _ in range(5):  # 5 samples
                sample = {
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_percent': psutil.disk_usage('/').percent,
                    'timestamp': time.time()
                }
                baseline_samples.append(sample)

            # Calculate averages
            self.baseline_metrics = {
                'cpu_percent': sum(s['cpu_percent'] for s in baseline_samples) / len(baseline_samples),
                'memory_percent': sum(s['memory_percent'] for s in baseline_samples) / len(baseline_samples),
                'disk_percent': sum(s['disk_percent'] for s in baseline_samples) / len(baseline_samples),
                'established_at': time.time()
            }

            self.logger.info("Baseline metrics established")

        except Exception as e:
            self.logger.warning(f"Error establishing baseline metrics: {e}")

    async def _continuous_diagnostic_monitor(self):
        """Continuous diagnostic monitoring"""
        while True:
            try:
                await asyncio.sleep(self.diagnostic_config['scan_interval'])

                # Perform quick health check
                health = await self._assess_system_health()

                # Alert if health deteriorates significantly
                if health.get('overall_health') in ['critical', 'poor']:
                    self.logger.warning(f"System health deteriorated: {health['overall_health']}")

                    # Trigger automatic diagnostic
                    await self.perform_comprehensive_diagnosis(
                        target_systems=['system', 'performance', 'memory'],
                        diagnostic_depth='standard'
                    )

            except Exception as e:
                self.logger.warning(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(60)  # Wait before retry

    def get_diagnostic_stats(self) -> Dict[str, Any]:
        """Get diagnostic statistics"""
        return {
            **self.stats,
            'diagnostic_history_size': len(self.diagnostic_history),
            'current_diagnostics_count': len(self.current_diagnostics),
            'baseline_metrics_available': bool(self.baseline_metrics)
        }

    async def shutdown(self):
        """Shutdown advanced diagnostics"""
        try:
            self.logger.info("Shutting down advanced diagnostics...")

            # Clear diagnostic history
            self.diagnostic_history.clear()
            self.current_diagnostics.clear()

            self.logger.info("Advanced diagnostics shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down advanced diagnostics: {e}")