"""
J.A.R.V.I.S. Performance Analyzer
Advanced performance analysis and bottleneck detection for AI-generated code
"""

import os
import time
import asyncio
import cProfile
import pstats
import io
import psutil
import threading
from typing import Dict, List, Any, Optional, Tuple
import tracemalloc
import logging

# Performance analysis imports
try:
    import memory_profiler
    import line_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False


class PerformanceAnalyzer:
    """
    Ultra-advanced performance analyzer that profiles code execution,
    identifies bottlenecks, and provides optimization recommendations
    """

    def __init__(self, development_engine):
        """
        Initialize performance analyzer

        Args:
            development_engine: Reference to self-development engine
        """
        self.engine = development_engine
        self.jarvis = development_engine.jarvis
        self.logger = logging.getLogger('JARVIS.PerformanceAnalyzer')

        # Performance baselines and thresholds
        self.performance_baselines = self._load_performance_baselines()
        self.performance_thresholds = self._load_performance_thresholds()

        # Analysis statistics
        self.stats = {
            'analyses_performed': 0,
            'bottlenecks_identified': 0,
            'optimizations_suggested': 0,
            'performance_improvements': 0,
            'memory_issues_detected': 0,
            'cpu_issues_detected': 0,
            'analysis_time': 0
        }

    async def initialize(self):
        """Initialize performance analyzer"""
        try:
            self.logger.info("Initializing performance analyzer...")

            # Start system monitoring
            self.system_monitor = SystemPerformanceMonitor()
            await self.system_monitor.start()

            self.logger.info("Performance analyzer initialized")

        except Exception as e:
            self.logger.error(f"Error initializing performance analyzer: {e}")
            raise

    async def analyze_performance(self,
                                code: str,
                                test_cases: List[Dict[str, Any]] = None,
                                analysis_depth: str = "comprehensive") -> Dict[str, Any]:
        """
        Comprehensive performance analysis

        Args:
            code: Code to analyze
            test_cases: Test cases for profiling
            analysis_depth: Analysis depth (basic, standard, comprehensive)

        Returns:
            Performance analysis results
        """
        start_time = time.time()

        try:
            self.logger.info(f"Performing {analysis_depth} performance analysis")

            # Generate test cases if not provided
            if not test_cases:
                test_cases = self._generate_test_cases(code)

            # Execute performance profiling
            profiling_results = await self._execute_performance_profiling(code, test_cases)

            # Analyze bottlenecks
            bottlenecks = await self._analyze_bottlenecks(profiling_results)

            # Memory analysis
            memory_analysis = await self._analyze_memory_usage(code, test_cases)

            # CPU analysis
            cpu_analysis = await self._analyze_cpu_usage(profiling_results)

            # Scalability analysis
            scalability_analysis = await self._analyze_scalability(code, test_cases)

            # Generate optimization recommendations
            recommendations = await self._generate_optimization_recommendations(
                bottlenecks, memory_analysis, cpu_analysis, scalability_analysis
            )

            # Calculate performance score
            performance_score = self._calculate_performance_score(
                profiling_results, bottlenecks, memory_analysis, cpu_analysis
            )

            analysis_time = time.time() - start_time
            self.stats['analysis_time'] += analysis_time
            self.stats['analyses_performed'] += 1

            result = {
                'analysis_depth': analysis_depth,
                'performance_score': performance_score,
                'profiling_results': profiling_results,
                'bottlenecks': bottlenecks,
                'memory_analysis': memory_analysis,
                'cpu_analysis': cpu_analysis,
                'scalability_analysis': scalability_analysis,
                'recommendations': recommendations,
                'test_cases_executed': len(test_cases),
                'analysis_time': analysis_time,
                'timestamp': time.time()
            }

            # Update statistics
            if bottlenecks:
                self.stats['bottlenecks_identified'] += len(bottlenecks)
            if memory_analysis.get('issues', []):
                self.stats['memory_issues_detected'] += len(memory_analysis['issues'])
            if cpu_analysis.get('issues', []):
                self.stats['cpu_issues_detected'] += len(cpu_analysis['issues'])
            if recommendations:
                self.stats['optimizations_suggested'] += len(recommendations)

            self.logger.info(f"Performance analysis completed with score: {performance_score:.2f}")
            return result

        except Exception as e:
            self.logger.error(f"Error in performance analysis: {e}")
            return {
                'error': str(e),
                'performance_score': 0.0,
                'analysis_time': time.time() - start_time
            }

    def _generate_test_cases(self, code: str) -> List[Dict[str, Any]]:
        """Generate test cases for performance profiling"""
        test_cases = []

        # Basic execution test
        test_cases.append({
            'name': 'basic_execution',
            'description': 'Basic code execution test',
            'input': {},
            'expected_output': None,
            'iterations': 10
        })

        # Try to identify functions in code
        import ast
        try:
            tree = ast.parse(code)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

            for func_name in functions[:3]:  # Limit to 3 functions
                test_cases.append({
                    'name': f'function_{func_name}',
                    'description': f'Test function {func_name}',
                    'input': {'function_name': func_name},
                    'expected_output': None,
                    'iterations': 5
                })

        except SyntaxError:
            pass

        return test_cases

    async def _execute_performance_profiling(self, code: str, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute performance profiling"""
        profiling_results = {
            'overall_stats': {},
            'function_stats': {},
            'test_case_results': []
        }

        try:
            # Profile each test case
            for test_case in test_cases:
                result = await self._profile_test_case(code, test_case)
                profiling_results['test_case_results'].append(result)

            # Aggregate results
            profiling_results['overall_stats'] = self._aggregate_profiling_results(profiling_results['test_case_results'])

            # Extract function-level statistics
            profiling_results['function_stats'] = self._extract_function_stats(profiling_results['test_case_results'])

        except Exception as e:
            self.logger.warning(f"Error in performance profiling: {e}")
            profiling_results['error'] = str(e)

        return profiling_results

    async def _profile_test_case(self, code: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Profile a specific test case"""
        result = {
            'test_case': test_case['name'],
            'execution_time': 0.0,
            'memory_usage': 0.0,
            'cpu_usage': 0.0,
            'profile_stats': {},
            'error': None
        }

        try:
            # Create profiling code
            profile_code = f"""
import cProfile
import pstats
import io
import time
import psutil
import os

# Test code
{code}

# Profiling function
def profile_execution():
    pr = cProfile.Profile()
    pr.enable()

    start_time = time.time()
    process = psutil.Process(os.getpid())

    try:
        # Execute test
        iterations = {test_case.get('iterations', 1)}
        for i in range(iterations):
            # Execute the code
            exec('''{code}''')

        execution_time = time.time() - start_time
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        cpu_usage = process.cpu_percent()

    except Exception as e:
        execution_time = time.time() - start_time
        memory_usage = 0
        cpu_usage = 0
        error = str(e)

    pr.disable()

    # Get profile stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10)
    profile_output = s.getvalue()

    return {{
        'execution_time': execution_time,
        'memory_usage': memory_usage,
        'cpu_usage': cpu_usage,
        'profile_stats': profile_output,
        'error': locals().get('error')
    }}

result = profile_execution()
"""

            # Execute profiling
            exec_result = await self._execute_code_safely(profile_code)

            if exec_result.get('success'):
                result.update(exec_result['result'])
            else:
                result['error'] = exec_result.get('error', 'Execution failed')

        except Exception as e:
            result['error'] = str(e)

        return result

    async def _execute_code_safely(self, code: str) -> Dict[str, Any]:
        """Execute code safely with timeout"""
        try:
            # Create a separate thread for execution
            result = {}

            def execute():
                try:
                    local_vars = {}
                    exec(code, {}, local_vars)
                    result['success'] = True
                    result['result'] = local_vars.get('result', {})
                except Exception as e:
                    result['success'] = False
                    result['error'] = str(e)

            thread = threading.Thread(target=execute)
            thread.start()
            thread.join(timeout=30)  # 30 second timeout

            if thread.is_alive():
                result['success'] = False
                result['error'] = 'Execution timed out'

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _aggregate_profiling_results(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate profiling results across test cases"""
        if not test_results:
            return {}

        # Calculate averages
        total_time = sum(r.get('execution_time', 0) for r in test_results)
        total_memory = sum(r.get('memory_usage', 0) for r in test_results)
        total_cpu = sum(r.get('cpu_usage', 0) for r in test_results)

        count = len(test_results)

        return {
            'avg_execution_time': total_time / count,
            'avg_memory_usage': total_memory / count,
            'avg_cpu_usage': total_cpu / count,
            'total_execution_time': total_time,
            'peak_memory_usage': max((r.get('memory_usage', 0) for r in test_results), default=0),
            'peak_cpu_usage': max((r.get('cpu_usage', 0) for r in test_results), default=0)
        }

    def _extract_function_stats(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract function-level statistics"""
        function_stats = {}

        for result in test_results:
            profile_stats = result.get('profile_stats', '')
            if profile_stats:
                # Parse profile output (simplified)
                lines = profile_stats.split('\n')
                for line in lines:
                    if line.strip() and not line.startswith(' '):
                        parts = line.split()
                        if len(parts) >= 6:
                            try:
                                function_name = parts[-1]
                                cumulative_time = float(parts[1])
                                calls = int(parts[0])

                                if function_name not in function_stats:
                                    function_stats[function_name] = {
                                        'calls': 0,
                                        'total_time': 0.0,
                                        'avg_time': 0.0
                                    }

                                function_stats[function_name]['calls'] += calls
                                function_stats[function_name]['total_time'] += cumulative_time

                            except (ValueError, IndexError):
                                continue

        # Calculate averages
        for func_name, stats in function_stats.items():
            if stats['calls'] > 0:
                stats['avg_time'] = stats['total_time'] / stats['calls']

        return function_stats

    async def _analyze_bottlenecks(self, profiling_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze performance bottlenecks"""
        bottlenecks = []

        # Analyze function statistics
        function_stats = profiling_results.get('function_stats', {})

        for func_name, stats in function_stats.items():
            avg_time = stats.get('avg_time', 0)
            calls = stats.get('calls', 0)

            # Check against thresholds
            if avg_time > self.performance_thresholds.get('function_time_threshold', 0.1):
                bottlenecks.append({
                    'type': 'slow_function',
                    'function': func_name,
                    'severity': 'high' if avg_time > 1.0 else 'medium',
                    'description': f"Function {func_name} is slow (avg {avg_time:.3f}s per call)",
                    'calls': calls,
                    'avg_time': avg_time,
                    'recommendation': 'Consider optimizing algorithm or caching results'
                })

        # Analyze overall statistics
        overall_stats = profiling_results.get('overall_stats', {})
        avg_execution_time = overall_stats.get('avg_execution_time', 0)

        if avg_execution_time > self.performance_thresholds.get('execution_time_threshold', 5.0):
            bottlenecks.append({
                'type': 'slow_execution',
                'severity': 'high',
                'description': f"Overall execution is slow (avg {avg_execution_time:.2f}s)",
                'avg_time': avg_execution_time,
                'recommendation': 'Profile code and optimize bottlenecks'
            })

        return bottlenecks

    async def _analyze_memory_usage(self, code: str, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze memory usage patterns"""
        memory_analysis = {
            'peak_usage': 0,
            'average_usage': 0,
            'memory_efficiency': 0.8,
            'issues': []
        }

        try:
            # Run memory profiling on test cases
            for test_case in test_cases:
                result = await self._profile_memory_test_case(code, test_case)

                memory_analysis['peak_usage'] = max(
                    memory_analysis['peak_usage'],
                    result.get('peak_memory', 0)
                )

                if result.get('memory_leaks'):
                    memory_analysis['issues'].extend(result['memory_leaks'])

            # Check against thresholds
            if memory_analysis['peak_usage'] > self.performance_thresholds.get('memory_threshold', 100):
                memory_analysis['issues'].append({
                    'type': 'high_memory_usage',
                    'severity': 'medium',
                    'description': f"High memory usage detected ({memory_analysis['peak_usage']:.1f} MB)",
                    'recommendation': 'Optimize memory usage or use streaming processing'
                })

        except Exception as e:
            self.logger.warning(f"Error in memory analysis: {e}")
            memory_analysis['issues'].append({
                'type': 'analysis_error',
                'severity': 'low',
                'description': f"Memory analysis failed: {e}"
            })

        return memory_analysis

    async def _profile_memory_test_case(self, code: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Profile memory usage for a test case"""
        result = {
            'peak_memory': 0,
            'memory_leaks': []
        }

        try:
            # Use tracemalloc for memory profiling
            tracemalloc.start()

            # Execute test case
            exec_result = await self._execute_code_safely(code)

            # Get memory statistics
            current, peak = tracemalloc.get_traced_memory()
            result['peak_memory'] = peak / 1024 / 1024  # MB

            # Check for potential memory leaks
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')

            for stat in top_stats[:5]:
                if stat.size > 1024 * 1024:  # 1MB
                    result['memory_leaks'].append({
                        'file': stat.traceback[0].filename,
                        'line': stat.traceback[0].lineno,
                        'size': stat.size / 1024 / 1024,  # MB
                        'count': stat.count
                    })

            tracemalloc.stop()

        except Exception as e:
            self.logger.warning(f"Error in memory profiling: {e}")

        return result

    async def _analyze_cpu_usage(self, profiling_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze CPU usage patterns"""
        cpu_analysis = {
            'avg_cpu_usage': 0,
            'peak_cpu_usage': 0,
            'cpu_efficiency': 0.8,
            'issues': []
        }

        try:
            # Analyze profiling results
            overall_stats = profiling_results.get('overall_stats', {})
            cpu_analysis['avg_cpu_usage'] = overall_stats.get('avg_cpu_usage', 0)
            cpu_analysis['peak_cpu_usage'] = overall_stats.get('peak_cpu_usage', 0)

            # Check for CPU-intensive operations
            if cpu_analysis['peak_cpu_usage'] > self.performance_thresholds.get('cpu_threshold', 80):
                cpu_analysis['issues'].append({
                    'type': 'high_cpu_usage',
                    'severity': 'medium',
                    'description': f"High CPU usage detected ({cpu_analysis['peak_cpu_usage']:.1f}%)",
                    'recommendation': 'Optimize CPU-intensive operations or use parallel processing'
                })

            # Analyze function-level CPU usage
            function_stats = profiling_results.get('function_stats', {})
            for func_name, stats in function_stats.items():
                if stats.get('avg_time', 0) > 0.5:  # More than 500ms per call
                    cpu_analysis['issues'].append({
                        'type': 'cpu_intensive_function',
                        'function': func_name,
                        'severity': 'low',
                        'description': f"Function {func_name} is CPU intensive",
                        'avg_time': stats['avg_time'],
                        'recommendation': 'Consider optimizing algorithm or caching'
                    })

        except Exception as e:
            self.logger.warning(f"Error in CPU analysis: {e}")

        return cpu_analysis

    async def _analyze_scalability(self, code: str, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze code scalability"""
        scalability_analysis = {
            'scalability_score': 0.7,
            'bottlenecks': [],
            'recommendations': []
        }

        try:
            # Analyze algorithmic complexity
            complexity = self._estimate_algorithmic_complexity(code)

            # Check for scalability issues
            if complexity.get('time_complexity') in ['O(n^2)', 'O(n^3)', 'O(2^n)']:
                scalability_analysis['bottlenecks'].append({
                    'type': 'algorithmic_complexity',
                    'severity': 'high',
                    'description': f"High algorithmic complexity: {complexity['time_complexity']}",
                    'recommendation': 'Consider more efficient algorithms'
                })
                scalability_analysis['scalability_score'] -= 0.3

            # Check for memory scaling issues
            if 'large_data' in code.lower():
                scalability_analysis['bottlenecks'].append({
                    'type': 'memory_scaling',
                    'severity': 'medium',
                    'description': 'Code may not scale well with large datasets',
                    'recommendation': 'Implement streaming or chunked processing'
                })
                scalability_analysis['scalability_score'] -= 0.2

        except Exception as e:
            self.logger.warning(f"Error in scalability analysis: {e}")

        return scalability_analysis

    def _estimate_algorithmic_complexity(self, code: str) -> Dict[str, Any]:
        """Estimate algorithmic complexity"""
        complexity = {
            'time_complexity': 'O(n)',
            'space_complexity': 'O(n)',
            'confidence': 0.5
        }

        # Simple heuristic-based analysis
        code_lower = code.lower()

        # Check for nested loops
        if code.count('for ') > 1 and 'for ' in code:
            nested_loops = code.count('for ') - 1
            if nested_loops >= 2:
                complexity['time_complexity'] = 'O(n^3)'
            elif nested_loops == 1:
                complexity['time_complexity'] = 'O(n^2)'

        # Check for exponential operations
        if '2**' in code or 'pow(2,' in code or 'fibonacci' in code_lower:
            complexity['time_complexity'] = 'O(2^n)'

        # Check for sorting operations
        if 'sort(' in code or 'sorted(' in code:
            complexity['time_complexity'] = 'O(n log n)'

        return complexity

    async def _generate_optimization_recommendations(self, bottlenecks: List[Dict[str, Any]],
                                                   memory_analysis: Dict[str, Any],
                                                   cpu_analysis: Dict[str, Any],
                                                   scalability_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations"""
        recommendations = []

        # Process bottlenecks
        for bottleneck in bottlenecks:
            recommendations.append({
                'type': 'bottleneck_optimization',
                'priority': 'high' if bottleneck.get('severity') == 'high' else 'medium',
                'target': bottleneck.get('function', 'general'),
                'description': bottleneck.get('description', ''),
                'recommendation': bottleneck.get('recommendation', ''),
                'expected_improvement': 0.2
            })

        # Process memory issues
        for issue in memory_analysis.get('issues', []):
            recommendations.append({
                'type': 'memory_optimization',
                'priority': 'medium',
                'target': 'memory_usage',
                'description': issue.get('description', ''),
                'recommendation': issue.get('recommendation', ''),
                'expected_improvement': 0.15
            })

        # Process CPU issues
        for issue in cpu_analysis.get('issues', []):
            recommendations.append({
                'type': 'cpu_optimization',
                'priority': 'medium',
                'target': 'cpu_usage',
                'description': issue.get('description', ''),
                'recommendation': issue.get('recommendation', ''),
                'expected_improvement': 0.1
            })

        # Process scalability issues
        for bottleneck in scalability_analysis.get('bottlenecks', []):
            recommendations.append({
                'type': 'scalability_optimization',
                'priority': 'high',
                'target': 'scalability',
                'description': bottleneck.get('description', ''),
                'recommendation': bottleneck.get('recommendation', ''),
                'expected_improvement': 0.25
            })

        # Remove duplicates and sort by priority
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            key = (rec['type'], rec['description'])
            if key not in seen:
                seen.add(key)
                unique_recommendations.append(rec)

        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        unique_recommendations.sort(key=lambda x: priority_order.get(x.get('priority', 'medium'), 1))

        return unique_recommendations

    def _calculate_performance_score(self, profiling_results: Dict[str, Any],
                                   bottlenecks: List[Dict[str, Any]],
                                   memory_analysis: Dict[str, Any],
                                   cpu_analysis: Dict[str, Any]) -> float:
        """Calculate overall performance score"""
        score = 100.0

        # Penalize for bottlenecks
        high_severity_bottlenecks = sum(1 for b in bottlenecks if b.get('severity') == 'high')
        score -= high_severity_bottlenecks * 15

        medium_severity_bottlenecks = sum(1 for b in bottlenecks if b.get('severity') == 'medium')
        score -= medium_severity_bottlenecks * 8

        # Penalize for memory issues
        memory_issues = len(memory_analysis.get('issues', []))
        score -= memory_issues * 10

        # Penalize for CPU issues
        cpu_issues = len(cpu_analysis.get('issues', []))
        score -= cpu_issues * 8

        # Bonus for good performance
        overall_stats = profiling_results.get('overall_stats', {})
        avg_time = overall_stats.get('avg_execution_time', 0)

        if avg_time < 1.0:  # Fast execution
            score += 10
        elif avg_time > 10.0:  # Slow execution
            score -= 20

        return max(0.0, min(100.0, score))

    def _load_performance_baselines(self) -> Dict[str, Any]:
        """Load performance baselines"""
        return {
            'function_call': {'avg_time': 0.001, 'memory_usage': 0.1},
            'list_comprehension': {'avg_time': 0.0005, 'memory_usage': 0.05},
            'dict_operations': {'avg_time': 0.0008, 'memory_usage': 0.08},
            'file_operations': {'avg_time': 0.01, 'memory_usage': 0.5}
        }

    def _load_performance_thresholds(self) -> Dict[str, Any]:
        """Load performance thresholds"""
        return {
            'function_time_threshold': 0.1,  # 100ms
            'execution_time_threshold': 5.0,  # 5 seconds
            'memory_threshold': 100,  # 100 MB
            'cpu_threshold': 80  # 80%
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance analysis statistics"""
        return {
            **self.stats,
            'avg_analysis_time': self.stats['analysis_time'] / max(1, self.stats['analyses_performed']),
            'bottleneck_ratio': self.stats['bottlenecks_identified'] / max(1, self.stats['analyses_performed'])
        }

    async def shutdown(self):
        """Shutdown performance analyzer"""
        try:
            self.logger.info("Shutting down performance analyzer...")

            # Stop system monitoring
            if hasattr(self, 'system_monitor'):
                await self.system_monitor.stop()

            self.logger.info("Performance analyzer shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down performance analyzer: {e}")


class SystemPerformanceMonitor:
    """Monitor system performance metrics"""

    def __init__(self):
        self.monitoring = False
        self.metrics_history = []
        self.monitor_thread = None

    async def start(self):
        """Start system monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    async def stop(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Collect system metrics
                metrics = {
                    'timestamp': time.time(),
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent
                }

                self.metrics_history.append(metrics)

                # Keep only recent history
                if len(self.metrics_history) > 100:
                    self.metrics_history.pop(0)

                time.sleep(5)  # Monitor every 5 seconds

            except Exception as e:
                logging.getLogger('JARVIS.PerformanceMonitor').warning(f"Monitoring error: {e}")
                time.sleep(10)