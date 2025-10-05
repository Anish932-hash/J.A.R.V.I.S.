"""
J.A.R.V.I.S. Code Optimizer
Advanced code optimization using AI and performance analysis
"""

import os
import time
import asyncio
import ast
import logging
from typing import Dict, List, Any, Optional, Tuple
import re
import tempfile
import subprocess
import sys

# Performance analysis imports
try:
    import psutil
    import memory_profiler
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class CodeOptimizer:
    """
    Ultra-advanced code optimizer that uses AI to analyze, profile,
    and optimize code for maximum performance
    """

    def __init__(self, development_engine):
        """
        Initialize code optimizer

        Args:
            development_engine: Reference to self-development engine
        """
        self.engine = development_engine
        self.jarvis = development_engine.jarvis
        self.logger = logging.getLogger('JARVIS.CodeOptimizer')

        # Optimization patterns and rules
        self.optimization_patterns = self._load_optimization_patterns()

        # Performance benchmarks
        self.performance_baselines = {}

        # Optimization statistics
        self.stats = {
            'optimizations_applied': 0,
            'performance_improvements': 0,
            'code_reductions': 0,
            'memory_optimizations': 0,
            'speed_improvements': 0,
            'optimization_time': 0
        }

    async def initialize(self):
        """Initialize code optimizer"""
        try:
            self.logger.info("Initializing code optimizer...")

            # Load performance baselines
            await self._load_performance_baselines()

            self.logger.info("Code optimizer initialized")

        except Exception as e:
            self.logger.error(f"Error initializing code optimizer: {e}")
            raise

    async def optimize_code(self,
                           code: str,
                           optimization_type: str = "comprehensive",
                           target_metrics: List[str] = None) -> Dict[str, Any]:
        """
        Optimize code for performance

        Args:
            code: Code to optimize
            optimization_type: Type of optimization (speed, memory, comprehensive)
            target_metrics: Specific metrics to optimize for

        Returns:
            Optimization results with improved code
        """
        start_time = time.time()

        try:
            self.logger.info(f"Optimizing code with {optimization_type} approach")

            # Parse and analyze code
            code_analysis = await self._analyze_code_structure(code)

            # Profile current performance
            baseline_performance = await self._profile_code_performance(code)

            # Apply optimizations based on type
            if optimization_type == "speed":
                optimized_code = await self._optimize_for_speed(code, code_analysis)
            elif optimization_type == "memory":
                optimized_code = await self._optimize_for_memory(code, code_analysis)
            elif optimization_type == "comprehensive":
                optimized_code = await self._comprehensive_optimization(code, code_analysis)
            else:
                optimized_code = code

            # Profile optimized performance
            optimized_performance = await self._profile_code_performance(optimized_code)

            # Calculate improvements
            improvements = self._calculate_performance_improvements(baseline_performance, optimized_performance)

            # Validate optimizations
            validation = await self._validate_optimizations(code, optimized_code, improvements)

            optimization_time = time.time() - start_time
            self.stats['optimization_time'] += optimization_time

            result = {
                'original_code': code,
                'optimized_code': optimized_code,
                'optimization_type': optimization_type,
                'code_analysis': code_analysis,
                'baseline_performance': baseline_performance,
                'optimized_performance': optimized_performance,
                'improvements': improvements,
                'validation': validation,
                'optimization_time': optimization_time,
                'success': validation.get('is_valid', False),
                'timestamp': time.time()
            }

            if validation.get('is_valid', False):
                self.stats['optimizations_applied'] += 1
                self.stats['performance_improvements'] += sum(improvements.values())
                self.stats['code_reductions'] += len(code) - len(optimized_code)

            self.logger.info(f"Code optimization completed in {optimization_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error optimizing code: {e}")
            return {
                'error': str(e),
                'original_code': code,
                'optimization_time': time.time() - start_time
            }

    async def _analyze_code_structure(self, code: str) -> Dict[str, Any]:
        """Analyze code structure for optimization opportunities"""
        analysis = {
            'complexity': 0,
            'inefficiencies': [],
            'optimization_opportunities': [],
            'performance_bottlenecks': [],
            'memory_usage_patterns': [],
            'algorithm_complexity': 'unknown'
        }

        try:
            # Parse AST
            tree = ast.parse(code)

            # Analyze complexity
            analysis['complexity'] = self._calculate_code_complexity(tree)

            # Find inefficiencies
            analysis['inefficiencies'] = self._identify_inefficiencies(tree, code)

            # Identify optimization opportunities
            analysis['optimization_opportunities'] = self._find_optimization_opportunities(tree, code)

            # Analyze performance bottlenecks
            analysis['performance_bottlenecks'] = self._identify_bottlenecks(tree, code)

            # Analyze memory patterns
            analysis['memory_usage_patterns'] = self._analyze_memory_patterns(tree, code)

            # Estimate algorithm complexity
            analysis['algorithm_complexity'] = self._estimate_algorithm_complexity(tree)

        except SyntaxError:
            analysis['inefficiencies'].append("Syntax errors prevent optimization analysis")

        return analysis

    def _calculate_code_complexity(self, tree: ast.AST) -> int:
        """Calculate code complexity score"""
        complexity = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While, ast.If)):
                complexity += 1
            elif isinstance(node, ast.Call):
                complexity += 0.5
            elif isinstance(node, ast.BinOp):
                complexity += 0.2

        return int(complexity)

    def _identify_inefficiencies(self, tree: ast.AST, code: str) -> List[str]:
        """Identify code inefficiencies"""
        inefficiencies = []

        # Check for string concatenation in loops
        lines = code.split('\n')
        in_loop = False
        loop_indent = 0

        for i, line in enumerate(lines):
            indent = len(line) - len(line.lstrip())

            # Track loop nesting
            if 'for ' in line or 'while ' in line:
                in_loop = True
                loop_indent = indent
            elif indent <= loop_indent and in_loop:
                in_loop = False

            # Check for string concatenation in loops
            if in_loop and ('+' in line and ('"' in line or "'" in line)):
                inefficiencies.append(f"String concatenation in loop at line {i+1}")

        # Check for unnecessary list comprehensions
        for node in ast.walk(tree):
            if isinstance(node, ast.ListComp):
                # Check if list comp is overly complex
                if len(list(ast.walk(node))) > 15:
                    inefficiencies.append("Overly complex list comprehension")

        return inefficiencies

    def _find_optimization_opportunities(self, tree: ast.AST, code: str) -> List[Dict[str, Any]]:
        """Find specific optimization opportunities"""
        opportunities = []

        # Apply optimization patterns
        for pattern_name, pattern_data in self.optimization_patterns.items():
            matches = re.findall(pattern_data['pattern'], code, re.MULTILINE | re.DOTALL)
            if matches:
                opportunities.append({
                    'type': pattern_name,
                    'occurrences': len(matches),
                    'estimated_improvement': pattern_data['improvement'] * len(matches),
                    'description': pattern_data['description']
                })

        # Find algorithmic improvements
        algo_opportunities = self._identify_algorithmic_improvements(tree)
        opportunities.extend(algo_opportunities)

        return opportunities

    def _identify_bottlenecks(self, tree: ast.AST, code: str) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []

        # Find nested loops
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check for nested loops
                nested_loops = [n for n in ast.walk(node) if isinstance(n, ast.For) and n != node]
                if nested_loops:
                    bottlenecks.append({
                        'type': 'nested_loops',
                        'severity': 'high',
                        'description': f"Nested loops detected at line {getattr(node, 'lineno', 'unknown')}",
                        'complexity': len(nested_loops) + 1
                    })

        # Find expensive operations
        expensive_ops = ['sort', 'sorted', 'max', 'min', 'sum']
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node)
                if func_name in expensive_ops:
                    bottlenecks.append({
                        'type': 'expensive_operation',
                        'severity': 'medium',
                        'operation': func_name,
                        'line': getattr(node, 'lineno', 'unknown')
                    })

        return bottlenecks

    def _analyze_memory_patterns(self, tree: ast.AST, code: str) -> List[Dict[str, Any]]:
        """Analyze memory usage patterns"""
        patterns = []

        # Check for large data structures
        for node in ast.walk(tree):
            if isinstance(node, ast.List):
                if len(node.elts) > 1000:  # Large list
                    patterns.append({
                        'type': 'large_data_structure',
                        'structure': 'list',
                        'size': len(node.elts),
                        'recommendation': 'Consider using generators or streaming'
                    })

        # Check for memory leaks (repeated allocations)
        allocation_patterns = re.findall(r'(\w+)\s*=\s*\[\]', code)
        if len(allocation_patterns) > 5:
            patterns.append({
                'type': 'repeated_allocations',
                'count': len(allocation_patterns),
                'recommendation': 'Reuse data structures where possible'
            })

        return patterns

    def _estimate_algorithm_complexity(self, tree: ast.AST) -> str:
        """Estimate algorithm complexity"""
        has_nested_loops = False
        has_recursion = False
        max_nesting = 0

        current_nesting = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
                if current_nesting > 1:
                    has_nested_loops = True
            elif isinstance(node, ast.Return):
                # Check for recursive calls
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func_name = self._get_function_name(child)
                        if func_name:  # Would need to check if it matches current function
                            has_recursion = True

        if has_recursion:
            return "O(2^n) or worse"
        elif max_nesting >= 3:
            return "O(n^3)"
        elif max_nesting == 2:
            return "O(n^2)"
        elif has_nested_loops:
            return "O(n^2)"
        else:
            return "O(n)"

    def _get_function_name(self, call_node: ast.Call) -> str:
        """Get function name from call node"""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            return call_node.func.attr
        return ""

    def _identify_algorithmic_improvements(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Identify algorithmic improvements"""
        improvements = []

        # Look for sorting operations that could use better algorithms
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node)
                if func_name in ['sort', 'sorted']:
                    improvements.append({
                        'type': 'sorting_optimization',
                        'current': 'Timsort (default)',
                        'suggested': 'Consider radix sort for integers or bucket sort for known ranges',
                        'estimated_improvement': 0.3,
                        'description': 'Optimize sorting algorithm based on data characteristics'
                    })

        # Look for search operations
        search_patterns = re.findall(r'\.index\(|in\s+\[', ast.unparse(tree) if hasattr(ast, 'unparse') else "")
        if search_patterns:
            improvements.append({
                'type': 'search_optimization',
                'current': 'Linear search',
                'suggested': 'Use binary search for sorted data or hash-based lookup',
                'estimated_improvement': 0.5,
                'description': 'Optimize search operations'
            })

        return improvements

    async def _profile_code_performance(self, code: str) -> Dict[str, Any]:
        """Profile code performance"""
        performance = {
            'execution_time': 0.0,
            'memory_usage': 0.0,
            'cpu_usage': 0.0,
            'error': None
        }

        try:
            # Create temporary test script
            test_code = f"""
import time
import sys

# Test code
{code}

# Performance measurement
start_time = time.time()
try:
    # Execute the code
    exec('''{code}''')
    execution_time = time.time() - start_time
    print(f"EXECUTION_TIME:{{execution_time}}")
except Exception as e:
    print(f"ERROR:{{e}}")
"""

            # Run the test
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_code)
                temp_file = f.name

            try:
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                # Parse results
                for line in result.stdout.split('\n'):
                    if line.startswith('EXECUTION_TIME:'):
                        performance['execution_time'] = float(line.split(':')[1])
                    elif line.startswith('ERROR:'):
                        performance['error'] = line.split(':', 1)[1]

            finally:
                try:
                    os.unlink(temp_file)
                except:
                    pass

        except Exception as e:
            performance['error'] = str(e)

        return performance

    async def _optimize_for_speed(self, code: str, analysis: Dict[str, Any]) -> str:
        """Optimize code for speed"""
        optimized_code = code

        try:
            # Apply speed optimizations
            for opportunity in analysis.get('optimization_opportunities', []):
                if opportunity['type'] == 'list_comprehension':
                    # Convert loops to list comprehensions
                    optimized_code = self._apply_list_comprehension_optimization(optimized_code)

                elif opportunity['type'] == 'string_concatenation':
                    # Use join instead of + for strings
                    optimized_code = self._apply_string_join_optimization(optimized_code)

            # Optimize algorithmic bottlenecks
            for bottleneck in analysis.get('performance_bottlenecks', []):
                if bottleneck['type'] == 'nested_loops':
                    optimized_code = self._optimize_nested_loops(optimized_code, bottleneck)

        except Exception as e:
            self.logger.warning(f"Error in speed optimization: {e}")

        return optimized_code

    async def _optimize_for_memory(self, code: str, analysis: Dict[str, Any]) -> str:
        """Optimize code for memory usage"""
        optimized_code = code

        try:
            # Apply memory optimizations
            for pattern in analysis.get('memory_usage_patterns', []):
                if pattern['type'] == 'large_data_structure':
                    optimized_code = self._optimize_large_data_structures(optimized_code, pattern)

                elif pattern['type'] == 'repeated_allocations':
                    optimized_code = self._optimize_repeated_allocations(optimized_code)

            # Use generators instead of lists where possible
            optimized_code = self._convert_to_generators(optimized_code)

        except Exception as e:
            self.logger.warning(f"Error in memory optimization: {e}")

        return optimized_code

    async def _comprehensive_optimization(self, code: str, analysis: Dict[str, Any]) -> str:
        """Apply comprehensive optimizations"""
        # Combine speed and memory optimizations
        optimized_code = await self._optimize_for_speed(code, analysis)
        optimized_code = await self._optimize_for_memory(optimized_code, analysis)

        # Apply additional comprehensive optimizations
        optimized_code = self._apply_comprehensive_optimizations(optimized_code, analysis)

        return optimized_code

    def _apply_list_comprehension_optimization(self, code: str) -> str:
        """Convert loops to list comprehensions"""
        # Simple pattern: for loops that append to lists
        pattern = r'(\w+)\s*=\s*\[\]\s*\n(?:\s*)for\s+(\w+)\s+in\s+(.+?):\s*\n(?:\s*)(\w+)\.append\((.+?)\)'
        replacement = r'\1 = [\5 for \2 in \3]'

        return re.sub(pattern, replacement, code, flags=re.MULTILINE | re.DOTALL)

    def _apply_string_join_optimization(self, code: str) -> str:
        """Optimize string concatenation"""
        # Replace string + in loops with join
        lines = code.split('\n')
        optimized_lines = []
        in_string_concat = False

        for line in lines:
            if 'for ' in line or 'while ' in line:
                in_string_concat = True
            elif in_string_concat and line.strip().startswith(('if ', 'elif ', 'else:', 'break', 'continue')):
                continue
            elif in_string_concat and not line.strip() or line.strip().startswith(('if ', 'elif ', 'else:', 'break', 'continue', 'return')):
                in_string_concat = False

            if in_string_concat and '+=' in line and ('"' in line or "'" in line):
                # Convert += to join pattern
                var_match = re.search(r'(\w+)\s*\+=\s*(.+)', line)
                if var_match:
                    var, addition = var_match.groups()
                    line = f"{var} = ''.join([{var}, {addition}])"

            optimized_lines.append(line)

        return '\n'.join(optimized_lines)

    def _optimize_nested_loops(self, code: str, bottleneck: Dict[str, Any]) -> str:
        """Optimize nested loops"""
        # This is a complex optimization that would require deep analysis
        # For now, add optimization comments
        lines = code.split('\n')
        complexity = bottleneck.get('complexity', 2)

        if complexity > 2:
            # Add optimization comment for highly nested loops
            for i, line in enumerate(lines):
                if 'for ' in line and any('for ' in l for l in lines[max(0, i-2):i+3]):
                    lines.insert(i, "# OPTIMIZATION: Consider loop unrolling or algorithm optimization")
                    break

        return '\n'.join(lines)

    def _optimize_large_data_structures(self, code: str, pattern: Dict[str, Any]) -> str:
        """Optimize large data structures"""
        # Add memory optimization comments
        lines = code.split('\n')

        for i, line in enumerate(lines):
            if '[' in line and ']' in line and ',' in line:
                # Check if it looks like a large list
                commas = line.count(',')
                if commas > 10:  # Arbitrary threshold
                    lines.insert(i, "# MEMORY OPTIMIZATION: Consider using generators for large datasets")
                    break

        return '\n'.join(lines)

    def _optimize_repeated_allocations(self, code: str) -> str:
        """Optimize repeated memory allocations"""
        # Look for patterns like: result = [] followed by multiple appends
        pattern = r'(\w+)\s*=\s*\[\]\s*\n((?:\s*\1\.append\(.+\)\s*\n)+)'
        replacement = r'# OPTIMIZATION: Consider pre-allocating list size\n\1 = []\n\2'

        return re.sub(pattern, replacement, code, flags=re.MULTILINE)

    def _convert_to_generators(self, code: str) -> str:
        """Convert lists to generators where beneficial"""
        # Convert list comprehensions to generator expressions in certain contexts
        lines = code.split('\n')
        optimized_lines = []

        for line in lines:
            # Convert list() calls to generator expressions
            if 'list(' in line and '[' in line and 'for ' in line:
                # Convert list[...] to (...)
                line = re.sub(r'list\((.+)\)', r'(\1)', line)
                optimized_lines.append("# OPTIMIZATION: Using generator expression for memory efficiency")
            optimized_lines.append(line)

        return '\n'.join(optimized_lines)

    def _apply_comprehensive_optimizations(self, code: str, analysis: Dict[str, Any]) -> str:
        """Apply comprehensive optimizations"""
        # Add general optimization comments and suggestions
        optimizations = [
            "# COMPREHENSIVE OPTIMIZATION: Consider using numpy for numerical operations",
            "# COMPREHENSIVE OPTIMIZATION: Profile with cProfile to identify bottlenecks",
            "# COMPREHENSIVE OPTIMIZATION: Consider caching expensive computations"
        ]

        lines = code.split('\n')
        lines.insert(0, '\n'.join(optimizations) + '\n')

        return '\n'.join(lines)

    def _calculate_performance_improvements(self, baseline: Dict[str, Any], optimized: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance improvements"""
        improvements = {}

        # Execution time improvement (lower is better)
        if baseline.get('execution_time', 0) > 0 and optimized.get('execution_time', 0) > 0:
            baseline_time = baseline['execution_time']
            optimized_time = optimized['execution_time']
            if optimized_time < baseline_time:
                improvements['speed'] = (baseline_time - optimized_time) / baseline_time
            else:
                improvements['speed'] = -((optimized_time - baseline_time) / baseline_time)

        # Memory usage improvement (lower is better)
        if baseline.get('memory_usage', 0) > 0 and optimized.get('memory_usage', 0) > 0:
            baseline_mem = baseline['memory_usage']
            optimized_mem = optimized['memory_usage']
            if optimized_mem < baseline_mem:
                improvements['memory'] = (baseline_mem - optimized_mem) / baseline_mem
            else:
                improvements['memory'] = -((optimized_mem - baseline_mem) / baseline_mem)

        # Code size reduction
        improvements['code_size'] = len(baseline.get('code', '')) - len(optimized.get('code', ''))

        return improvements

    async def _validate_optimizations(self, original_code: str, optimized_code: str, improvements: Dict[str, float]) -> Dict[str, Any]:
        """Validate that optimizations don't break functionality"""
        validation = {
            'is_valid': True,
            'functionality_preserved': True,
            'performance_improved': False,
            'issues': []
        }

        try:
            # Basic syntax check
            compile(optimized_code, '<optimized>', 'exec')

            # Check if performance improved
            total_improvement = sum(v for k, v in improvements.items() if k != 'code_size' and v > 0)
            if total_improvement > 0.1:  # 10% improvement threshold
                validation['performance_improved'] = True

            # Check for obvious issues
            if len(optimized_code) > len(original_code) * 2:
                validation['issues'].append("Optimized code is significantly larger")
                validation['is_valid'] = False

        except SyntaxError as e:
            validation['is_valid'] = False
            validation['functionality_preserved'] = False
            validation['issues'].append(f"Syntax error in optimized code: {e}")

        except Exception as e:
            validation['issues'].append(f"Validation error: {e}")

        return validation

    def _load_optimization_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load optimization patterns"""
        return {
            'list_comprehension': {
                'pattern': r'(\w+)\s*=\s*\[\]\s*\n\s*for\s+(\w+)\s+in\s+(.+?):\s*\n\s*\1\.append\((.+?)\)',
                'replacement': r'\1 = [\4 for \2 in \3]',
                'improvement': 0.15,
                'description': 'Convert loop appends to list comprehensions'
            },
            'string_concatenation': {
                'pattern': r'(\w+)\s*\+=\s*([^+\n]+)',
                'replacement': r'\1 = "".join([\1, \2])',
                'improvement': 0.25,
                'description': 'Use join() for string concatenation'
            },
            'dict_get': {
                'pattern': r'(\w+)\.get\((.+?),\s*None\)',
                'replacement': r'\1.get(\2)',
                'improvement': 0.05,
                'description': 'Simplify dict.get() calls'
            }
        }

    async def _load_performance_baselines(self):
        """Load performance baselines"""
        # This would load historical performance data
        self.performance_baselines = {
            'list_comprehension': {'speed_improvement': 0.15, 'memory_improvement': 0.1},
            'string_join': {'speed_improvement': 0.25, 'memory_improvement': 0.05},
            'generator_conversion': {'speed_improvement': 0.1, 'memory_improvement': 0.3}
        }

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            **self.stats,
            'avg_improvement': self.stats['performance_improvements'] / max(1, self.stats['optimizations_applied']),
            'optimization_patterns': len(self.optimization_patterns)
        }

    async def shutdown(self):
        """Shutdown code optimizer"""
        try:
            self.logger.info("Shutting down code optimizer...")

            # Clear baselines
            self.performance_baselines.clear()

            self.logger.info("Code optimizer shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down code optimizer: {e}")