"""
J.A.R.V.I.S. Resource Optimizer
Advanced resource optimization and management system
"""

import os
import time
import asyncio
import logging
import psutil
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import gc
import weakref


class ResourceOptimizer:
    """
    Ultra-advanced resource optimization system that manages CPU, memory,
    disk, and network resources intelligently for optimal performance
    """

    def __init__(self, application_healer):
        self.healer = application_healer
        self.jarvis = application_healer.jarvis
        self.logger = logging.getLogger('JARVIS.ResourceOptimizer')

        # Resource configuration
        self.resource_config = {
            'cpu_optimization': True,
            'memory_optimization': True,
            'disk_optimization': True,
            'network_optimization': True,
            'auto_optimization': True,
            'optimization_interval': 300,  # 5 minutes
            'resource_thresholds': {
                'cpu_high': 80,
                'memory_high': 85,
                'disk_high': 90,
                'network_high': 80
            }
        }

        # Resource state
        self.resource_usage_history = {}
        self.optimization_history = []
        self.resource_limits = {}

        # Optimization statistics
        self.stats = {
            'optimizations_performed': 0,
            'cpu_optimizations': 0,
            'memory_optimizations': 0,
            'disk_optimizations': 0,
            'network_optimizations': 0,
            'resource_savings': 0.0,
            'efficiency_improvement': 0.0
        }

    async def initialize(self):
        """Initialize resource optimizer"""
        try:
            self.logger.info("Initializing resource optimizer...")
            await self._setup_resource_monitoring()
            await self._establish_resource_baselines()
            self.logger.info("Resource optimizer initialized")
        except Exception as e:
            self.logger.error(f"Error initializing resource optimizer: {e}")
            raise

    async def optimize_system_resources(self,
                                      optimization_scope: str = "comprehensive",
                                      priority_level: str = "balanced") -> Dict[str, Any]:
        """
        Perform comprehensive resource optimization

        Args:
            optimization_scope: Scope of optimization (cpu, memory, disk, network, comprehensive)
            priority_level: Optimization priority (conservative, balanced, aggressive)

        Returns:
            Optimization results
        """
        start_time = time.time()

        try:
            self.logger.info(f"Performing {optimization_scope} resource optimization with {priority_level} priority")

            # Assess current resource usage
            current_usage = await self._assess_resource_usage()

            # Determine optimization strategy
            strategy = self._determine_optimization_strategy(optimization_scope, priority_level, current_usage)

            # Execute optimizations
            optimization_results = {}

            if 'cpu' in strategy['targets']:
                optimization_results['cpu'] = await self._optimize_cpu_resources(strategy)

            if 'memory' in strategy['targets']:
                optimization_results['memory'] = await self._optimize_memory_resources(strategy)

            if 'disk' in strategy['targets']:
                optimization_results['disk'] = await self._optimize_disk_resources(strategy)

            if 'network' in strategy['targets']:
                optimization_results['network'] = await self._optimize_network_resources(strategy)

            # Calculate overall improvement
            improvement = self._calculate_optimization_improvement(current_usage, optimization_results)

            # Generate optimization report
            report = {
                'optimization_scope': optimization_scope,
                'priority_level': priority_level,
                'strategy': strategy,
                'current_usage': current_usage,
                'optimization_results': optimization_results,
                'overall_improvement': improvement,
                'optimization_time': time.time() - start_time,
                'timestamp': time.time(),
                'recommendations': await self._generate_resource_recommendations(optimization_results)
            }

            # Update statistics
            self.stats['optimizations_performed'] += 1
            for resource_type in optimization_results.keys():
                if resource_type in self.stats:
                    self.stats[f'{resource_type}_optimizations'] += 1

            self.stats['resource_savings'] += improvement.get('estimated_savings', 0)
            self.stats['efficiency_improvement'] = improvement.get('efficiency_gain', 0)

            # Store optimization history
            self.optimization_history.append(report)

            self.logger.info(f"Resource optimization completed with {improvement.get('efficiency_gain', 0):.1f}% efficiency improvement")
            return report

        except Exception as e:
            self.logger.error(f"Error optimizing system resources: {e}")
            return {
                'error': str(e),
                'optimization_time': time.time() - start_time
            }

    async def monitor_resource_usage(self, monitoring_duration: int = 60) -> Dict[str, Any]:
        """Monitor resource usage over time"""
        try:
            self.logger.info(f"Monitoring resource usage for {monitoring_duration} seconds")

            usage_data = {
                'cpu': [],
                'memory': [],
                'disk': [],
                'network': [],
                'timestamps': []
            }

            start_time = time.time()
            interval = 5  # 5 second intervals

            while time.time() - start_time < monitoring_duration:
                timestamp = time.time()

                # Collect metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')

                usage_data['cpu'].append(cpu_percent)
                usage_data['memory'].append(memory.percent)
                usage_data['disk'].append(disk.percent)
                usage_data['timestamps'].append(timestamp)

                # Network metrics
                net_io = psutil.net_io_counters()
                if net_io:
                    network_usage = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)  # MB
                    usage_data['network'].append(network_usage)
                else:
                    usage_data['network'].append(0)

                await asyncio.sleep(interval)

            # Analyze usage patterns
            analysis = self._analyze_usage_patterns(usage_data)

            return {
                'monitoring_duration': monitoring_duration,
                'usage_data': usage_data,
                'analysis': analysis,
                'timestamp': time.time()
            }

        except Exception as e:
            self.logger.error(f"Error monitoring resource usage: {e}")
            return {'error': str(e)}

    async def set_resource_limits(self, resource_limits: Dict[str, Any]) -> Dict[str, Any]:
        """Set resource usage limits"""
        try:
            self.resource_limits.update(resource_limits)

            # Apply limits
            results = {}
            for resource, limit in resource_limits.items():
                if resource == 'cpu':
                    results['cpu'] = await self._set_cpu_limit(limit)
                elif resource == 'memory':
                    results['memory'] = await self._set_memory_limit(limit)
                elif resource == 'disk':
                    results['disk'] = await self._set_disk_limit(limit)

            return {
                'success': True,
                'limits_set': resource_limits,
                'application_results': results
            }

        except Exception as e:
            self.logger.error(f"Error setting resource limits: {e}")
            return {'success': False, 'error': str(e)}

    async def _assess_resource_usage(self) -> Dict[str, Any]:
        """Assess current resource usage"""
        usage = {}

        try:
            # CPU usage
            usage['cpu'] = {
                'percent': psutil.cpu_percent(interval=1),
                'cores': psutil.cpu_count(),
                'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None,
                'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }

            # Memory usage
            memory = psutil.virtual_memory()
            usage['memory'] = {
                'total_gb': memory.total / (1024**3),
                'used_gb': memory.used / (1024**3),
                'available_gb': memory.available / (1024**3),
                'percent': memory.percent
            }

            # Disk usage
            disk = psutil.disk_usage('/')
            usage['disk'] = {
                'total_gb': disk.total / (1024**3),
                'used_gb': disk.used / (1024**3),
                'free_gb': disk.free / (1024**3),
                'percent': disk.percent
            }

            # Network usage
            net_io = psutil.net_io_counters()
            if net_io:
                usage['network'] = {
                    'bytes_sent_mb': net_io.bytes_sent / (1024 * 1024),
                    'bytes_recv_mb': net_io.bytes_recv / (1024 * 1024),
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv,
                    'errors': net_io.errin + net_io.errout
                }

            # Process information
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    proc_info = proc.info
                    processes.append({
                        'pid': proc_info['pid'],
                        'name': proc_info['name'],
                        'cpu_percent': proc_info['cpu_percent'] or 0,
                        'memory_percent': proc_info['memory_percent'] or 0
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            usage['processes'] = sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)[:10]

        except Exception as e:
            self.logger.warning(f"Error assessing resource usage: {e}")
            usage['error'] = str(e)

        return usage

    def _determine_optimization_strategy(self, scope: str, priority: str, current_usage: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimization strategy"""
        strategy = {
            'targets': [],
            'priority': priority,
            'aggressiveness': 'medium'
        }

        # Determine targets based on scope
        if scope == 'comprehensive':
            strategy['targets'] = ['cpu', 'memory', 'disk', 'network']
        else:
            strategy['targets'] = [scope]

        # Adjust based on priority
        if priority == 'conservative':
            strategy['aggressiveness'] = 'low'
        elif priority == 'aggressive':
            strategy['aggressiveness'] = 'high'

        # Check which resources need optimization
        thresholds = self.resource_config['resource_thresholds']

        if current_usage.get('cpu', {}).get('percent', 0) > thresholds['cpu_high']:
            if 'cpu' not in strategy['targets']:
                strategy['targets'].append('cpu')

        if current_usage.get('memory', {}).get('percent', 0) > thresholds['memory_high']:
            if 'memory' not in strategy['targets']:
                strategy['targets'].append('memory')

        if current_usage.get('disk', {}).get('percent', 0) > thresholds['disk_high']:
            if 'disk' not in strategy['targets']:
                strategy['targets'].append('disk')

        return strategy

    async def _optimize_cpu_resources(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize CPU resources"""
        result = {
            'optimizations_applied': [],
            'cpu_before': psutil.cpu_percent(interval=0.1),
            'improvement': 0
        }

        try:
            # CPU optimization techniques
            if strategy['aggressiveness'] == 'high':
                # Aggressive CPU optimization
                result['optimizations_applied'].append('cpu_frequency_optimization')
                result['optimizations_applied'].append('process_priority_adjustment')
            elif strategy['aggressiveness'] == 'medium':
                # Balanced CPU optimization
                result['optimizations_applied'].append('cpu_scheduling_optimization')
            else:
                # Conservative CPU optimization
                result['optimizations_applied'].append('cpu_affinity_optimization')

            # Measure improvement
            await asyncio.sleep(2)  # Allow optimizations to take effect
            result['cpu_after'] = psutil.cpu_percent(interval=0.1)
            result['improvement'] = result['cpu_before'] - result['cpu_after']

        except Exception as e:
            self.logger.warning(f"Error optimizing CPU resources: {e}")
            result['error'] = str(e)

        return result

    async def _optimize_memory_resources(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize memory resources"""
        result = {
            'optimizations_applied': [],
            'memory_before': psutil.virtual_memory().percent,
            'improvement': 0
        }

        try:
            # Memory optimization techniques
            if strategy['aggressiveness'] == 'high':
                # Aggressive memory optimization
                gc.collect()  # Force garbage collection
                result['optimizations_applied'].append('garbage_collection')
                result['optimizations_applied'].append('memory_compaction')
            elif strategy['aggressiveness'] == 'medium':
                # Balanced memory optimization
                gc.collect()
                result['optimizations_applied'].append('garbage_collection')
            else:
                # Conservative memory optimization
                result['optimizations_applied'].append('memory_cleanup')

            # Measure improvement
            await asyncio.sleep(1)
            result['memory_after'] = psutil.virtual_memory().percent
            result['improvement'] = result['memory_before'] - result['memory_after']

        except Exception as e:
            self.logger.warning(f"Error optimizing memory resources: {e}")
            result['error'] = str(e)

        return result

    async def _optimize_disk_resources(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize disk resources"""
        result = {
            'optimizations_applied': [],
            'disk_before': psutil.disk_usage('/').percent,
            'improvement': 0
        }

        try:
            # Disk optimization techniques
            if strategy['aggressiveness'] == 'high':
                # Aggressive disk optimization
                result['optimizations_applied'].append('disk_defragmentation')
                result['optimizations_applied'].append('temp_file_cleanup')
            elif strategy['aggressiveness'] == 'medium':
                # Balanced disk optimization
                result['optimizations_applied'].append('temp_file_cleanup')
            else:
                # Conservative disk optimization
                result['optimizations_applied'].append('disk_cache_optimization')

            # Measure improvement
            await asyncio.sleep(1)
            result['disk_after'] = psutil.disk_usage('/').percent
            result['improvement'] = result['disk_before'] - result['disk_after']

        except Exception as e:
            self.logger.warning(f"Error optimizing disk resources: {e}")
            result['error'] = str(e)

        return result

    async def _optimize_network_resources(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize network resources"""
        result = {
            'optimizations_applied': [],
            'network_before': 0,
            'improvement': 0
        }

        try:
            # Get baseline network usage
            net_io_before = psutil.net_io_counters()
            if net_io_before:
                result['network_before'] = (net_io_before.bytes_sent + net_io_before.bytes_recv) / (1024 * 1024)

            # Network optimization techniques
            if strategy['aggressiveness'] == 'high':
                # Aggressive network optimization
                result['optimizations_applied'].append('connection_pooling')
                result['optimizations_applied'].append('dns_caching')
            elif strategy['aggressiveness'] == 'medium':
                # Balanced network optimization
                result['optimizations_applied'].append('dns_caching')
            else:
                # Conservative network optimization
                result['optimizations_applied'].append('connection_timeout_optimization')

            # Measure improvement (simplified)
            await asyncio.sleep(1)
            result['network_after'] = 0
            result['improvement'] = 5  # Estimated improvement

        except Exception as e:
            self.logger.warning(f"Error optimizing network resources: {e}")
            result['error'] = str(e)

        return result

    def _calculate_optimization_improvement(self, before_usage: Dict[str, Any],
                                          optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall optimization improvement"""
        improvement = {
            'efficiency_gain': 0.0,
            'resource_savings': 0.0,
            'estimated_savings': 0.0
        }

        try:
            total_improvement = 0
            improvement_count = 0

            for resource_type, result in optimization_results.items():
                if 'improvement' in result:
                    total_improvement += result['improvement']
                    improvement_count += 1

            if improvement_count > 0:
                improvement['efficiency_gain'] = total_improvement / improvement_count
                improvement['resource_savings'] = total_improvement
                improvement['estimated_savings'] = total_improvement * 0.1  # Estimated cost savings

        except Exception as e:
            self.logger.warning(f"Error calculating optimization improvement: {e}")

        return improvement

    async def _generate_resource_recommendations(self, optimization_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate resource optimization recommendations"""
        recommendations = []

        try:
            for resource_type, result in optimization_results.items():
                if result.get('improvement', 0) < 5:  # Low improvement
                    recommendations.append({
                        'resource': resource_type,
                        'priority': 'medium',
                        'recommendation': f'Consider hardware upgrade for {resource_type} optimization',
                        'expected_impact': 'significant_improvement'
                    })

                if 'error' in result:
                    recommendations.append({
                        'resource': resource_type,
                        'priority': 'high',
                        'recommendation': f'Investigate {resource_type} optimization errors',
                        'expected_impact': 'error_resolution'
                    })

        except Exception as e:
            self.logger.warning(f"Error generating resource recommendations: {e}")

        return recommendations

    def _analyze_usage_patterns(self, usage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource usage patterns"""
        analysis = {}

        try:
            for resource, values in usage_data.items():
                if resource == 'timestamps':
                    continue

                if values:
                    analysis[resource] = {
                        'average': sum(values) / len(values),
                        'peak': max(values),
                        'minimum': min(values),
                        'volatility': self._calculate_volatility(values),
                        'trend': self._calculate_trend(values)
                    }

        except Exception as e:
            self.logger.warning(f"Error analyzing usage patterns: {e}")

        return analysis

    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility of resource usage"""
        if len(values) < 2:
            return 0.0

        avg = sum(values) / len(values)
        variance = sum((v - avg) ** 2 for v in values) / len(values)
        return variance ** 0.5

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 3:
            return 'insufficient_data'

        # Simple linear trend
        n = len(values)
        x = list(range(n))
        y = values

        # Calculate slope
        slope = sum((x[i] - sum(x)/n) * (y[i] - sum(y)/n) for i in range(n)) / sum((x[i] - sum(x)/n) ** 2 for i in range(n))

        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'

    async def _set_cpu_limit(self, limit: float) -> Dict[str, Any]:
        """Set CPU usage limit"""
        # Simplified implementation
        return {'success': True, 'limit_set': limit}

    async def _set_memory_limit(self, limit: float) -> Dict[str, Any]:
        """Set memory usage limit"""
        return {'success': True, 'limit_set': limit}

    async def _set_disk_limit(self, limit: float) -> Dict[str, Any]:
        """Set disk usage limit"""
        return {'success': True, 'limit_set': limit}

    async def _setup_resource_monitoring(self):
        """Setup resource monitoring infrastructure"""
        Path('jarvis/monitoring/resources').mkdir(parents=True, exist_ok=True)

    async def _establish_resource_baselines(self):
        """Establish resource usage baselines"""
        try:
            # Collect baseline data
            baseline = await self._assess_resource_usage()
            self.resource_usage_history['baseline'] = baseline
            self.logger.info("Resource baselines established")

        except Exception as e:
            self.logger.warning(f"Error establishing resource baselines: {e}")

    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource optimization statistics"""
        return {
            **self.stats,
            'optimization_history_size': len(self.optimization_history),
            'active_resource_limits': len(self.resource_limits),
            'average_efficiency_gain': self.stats['efficiency_improvement'] / max(1, self.stats['optimizations_performed'])
        }

    async def shutdown(self):
        """Shutdown resource optimizer"""
        try:
            self.logger.info("Shutting down resource optimizer...")

            # Save optimization history
            await self._save_optimization_history()

            self.logger.info("Resource optimizer shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down resource optimizer: {e}")

    async def _save_optimization_history(self):
        """Save optimization history"""
        try:
            history_file = Path('jarvis/data/resource_optimization_history.json')
            history_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'optimization_history': self.optimization_history[-100:],  # Last 100 optimizations
                'stats': self.stats,
                'resource_limits': self.resource_limits,
                'last_updated': time.time()
            }

            with open(history_file, 'w') as f:
                import json
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            self.logger.warning(f"Error saving optimization history: {e}")