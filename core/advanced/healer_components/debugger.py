"""
J.A.R.V.I.S. Advanced Debugger
Comprehensive debugging and error analysis system with AI-powered root cause analysis
"""

import os
import sys
import time
import ast
import traceback
import inspect
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    SYNTAX = "syntax"
    RUNTIME = "runtime"
    LOGIC = "logic"
    RESOURCE = "resource"
    NETWORK = "network"
    SECURITY = "security"
    PERFORMANCE = "performance"
    CONFIGURATION = "configuration"


@dataclass
class DebugSession:
    """Represents a debugging session"""
    session_id: str
    error_info: Dict[str, Any]
    start_time: float
    stack_trace: List[Dict[str, Any]] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    breakpoints: List[Dict[str, Any]] = field(default_factory=list)
    execution_path: List[str] = field(default_factory=list)
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"


@dataclass
class CodeLocation:
    """Represents a location in code"""
    file_path: str
    line_number: int
    function_name: str = ""
    class_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "function_name": self.function_name,
            "class_name": self.class_name
        }


class CallGraphAnalyzer:
    """Analyzes function call graphs for debugging"""

    def __init__(self):
        self.call_graph = {}
        self.reverse_call_graph = {}

    def build_call_graph(self, code: str, file_path: str) -> Dict[str, Any]:
        """Build call graph from code"""
        try:
            tree = ast.parse(code)

            # Extract function definitions and calls
            functions = {}
            calls = {}

            class CallVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.current_function = None
                    self.functions = {}
                    self.calls = {}

                def visit_FunctionDef(self, node):
                    self.functions[node.name] = {
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args],
                        "calls": []
                    }
                    old_function = self.current_function
                    self.current_function = node.name
                    self.generic_visit(node)
                    self.current_function = old_function

                def visit_Call(self, node):
                    if self.current_function and isinstance(node.func, ast.Name):
                        if self.current_function not in self.calls:
                            self.calls[self.current_function] = []
                        self.calls[self.current_function].append(node.func.id)
                    self.generic_visit(node)

            visitor = CallVisitor()
            visitor.visit(tree)

            return {
                "functions": visitor.functions,
                "calls": visitor.calls,
                "file_path": file_path
            }

        except Exception as e:
            return {"error": str(e)}

    def analyze_execution_path(self, stack_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze execution path from stack trace"""
        path_analysis = {
            "call_chain": [],
            "potential_issues": [],
            "complexity_score": 0
        }

        for frame in stack_trace:
            path_analysis["call_chain"].append({
                "function": frame.get("function", "unknown"),
                "file": frame.get("file", "unknown"),
                "line": frame.get("line", 0)
            })

        # Analyze for potential issues
        if len(path_analysis["call_chain"]) > 50:
            path_analysis["potential_issues"].append("Deep call stack - possible recursion")

        # Calculate complexity
        unique_functions = len(set(frame["function"] for frame in path_analysis["call_chain"]))
        path_analysis["complexity_score"] = min(100, len(path_analysis["call_chain"]) * unique_functions / 10)

        return path_analysis


class VariableTracer:
    """Traces variable values and changes"""

    def __init__(self):
        self.variable_history = {}
        self.watch_list = set()

    def trace_variable(self, name: str, value: Any, location: CodeLocation):
        """Trace a variable's value at a location"""
        if name not in self.variable_history:
            self.variable_history[name] = []

        self.variable_history[name].append({
            "value": str(value),
            "type": type(value).__name__,
            "location": location.to_dict(),
            "timestamp": time.time()
        })

        # Keep only recent history
        if len(self.variable_history[name]) > 100:
            self.variable_history[name].pop(0)

    def analyze_variable_changes(self, variable_name: str) -> Dict[str, Any]:
        """Analyze changes in a variable over time"""
        if variable_name not in self.variable_history:
            return {"error": "Variable not found in history"}

        history = self.variable_history[variable_name]

        analysis = {
            "total_changes": len(history),
            "unique_values": len(set(entry["value"] for entry in history)),
            "type_changes": len(set(entry["type"] for entry in history)),
            "change_frequency": 0,
            "locations": []
        }

        if len(history) > 1:
            time_span = history[-1]["timestamp"] - history[0]["timestamp"]
            analysis["change_frequency"] = len(history) / max(time_span, 1)

        analysis["locations"] = [entry["location"] for entry in history[-10:]]  # Last 10 locations

        return analysis

    def detect_anomalies(self, variable_name: str) -> List[str]:
        """Detect anomalies in variable behavior"""
        anomalies = []

        if variable_name not in self.variable_history:
            return anomalies

        history = self.variable_history[variable_name]

        if len(history) < 3:
            return anomalies

        # Check for type instability
        types = [entry["type"] for entry in history]
        if len(set(types)) > 3:
            anomalies.append(f"Variable {variable_name} changes type frequently")

        # Check for rapid changes
        if len(history) > 10:
            recent_changes = sum(1 for i in range(1, len(history))
                                if history[i]["value"] != history[i-1]["value"])
            if recent_changes > len(history) * 0.8:
                anomalies.append(f"Variable {variable_name} changes very frequently")

        return anomalies


class PatternMatcher:
    """Matches error patterns for diagnosis"""

    def __init__(self):
        self.error_patterns = {
            "recursion": {
                "indicators": ["maximum recursion depth", "RecursionError", "recursive"],
                "solutions": ["Check for infinite recursion", "Add base case", "Use iterative approach"]
            },
            "memory_leak": {
                "indicators": ["MemoryError", "out of memory", "memory leak"],
                "solutions": ["Check for circular references", "Use weak references", "Implement proper cleanup"]
            },
            "type_error": {
                "indicators": ["TypeError", "unsupported operand", "can't convert"],
                "solutions": ["Check variable types", "Add type checking", "Use type hints"]
            },
            "key_error": {
                "indicators": ["KeyError", "key not found", "missing key"],
                "solutions": ["Check dictionary keys", "Use .get() method", "Add key validation"]
            },
            "attribute_error": {
                "indicators": ["AttributeError", "has no attribute", "object has no"],
                "solutions": ["Check object attributes", "Verify object initialization", "Use hasattr()"]
            },
            "import_error": {
                "indicators": ["ImportError", "ModuleNotFoundError", "No module named"],
                "solutions": ["Check module installation", "Verify import path", "Check Python path"]
            },
            "permission_error": {
                "indicators": ["PermissionError", "Access denied", "permission denied"],
                "solutions": ["Check file permissions", "Run with appropriate privileges", "Check user rights"]
            }
        }

    def match_error(self, error_message: str, stack_trace: str = "") -> Dict[str, Any]:
        """Match error against known patterns"""
        error_text = (error_message + " " + stack_trace).lower()

        matches = []
        for pattern_name, pattern_data in self.error_patterns.items():
            score = 0
            for indicator in pattern_data["indicators"]:
                if indicator.lower() in error_text:
                    score += 1

            if score > 0:
                matches.append({
                    "pattern": pattern_name,
                    "confidence": score / len(pattern_data["indicators"]),
                    "solutions": pattern_data["solutions"]
                })

        # Sort by confidence
        matches.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "matches": matches[:3],  # Top 3 matches
            "best_match": matches[0] if matches else None
        }


class RootCauseAnalyzer:
    """Analyzes root causes of errors"""

    def __init__(self):
        self.pattern_matcher = PatternMatcher()
        self.call_graph_analyzer = CallGraphAnalyzer()
        self.variable_tracer = VariableTracer()

    def analyze_root_cause(self, error_info: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive root cause analysis"""
        analysis = {
            "primary_cause": "",
            "contributing_factors": [],
            "confidence": 0.0,
            "recommended_actions": [],
            "prevention_measures": []
        }

        error_message = error_info.get("message", "")
        stack_trace = error_info.get("traceback", "")

        # Pattern matching
        pattern_match = self.pattern_matcher.match_error(error_message, stack_trace)

        if pattern_match["best_match"]:
            best_match = pattern_match["best_match"]
            analysis["primary_cause"] = best_match["pattern"]
            analysis["confidence"] = best_match["confidence"]
            analysis["recommended_actions"] = best_match["solutions"]

        # Stack trace analysis
        if stack_trace:
            stack_analysis = self._analyze_stack_trace(stack_trace)
            analysis["contributing_factors"].extend(stack_analysis.get("issues", []))

        # Code analysis
        if "file_path" in error_info and "line_number" in error_info:
            code_analysis = self._analyze_error_location(
                error_info["file_path"],
                error_info["line_number"]
            )
            analysis["contributing_factors"].extend(code_analysis.get("issues", []))

        # Context analysis
        context_issues = self._analyze_error_context(context)
        analysis["contributing_factors"].extend(context_issues)

        # Generate prevention measures
        analysis["prevention_measures"] = self._generate_prevention_measures(analysis)

        return analysis

    def _analyze_stack_trace(self, stack_trace: str) -> Dict[str, Any]:
        """Analyze stack trace for issues"""
        issues = []

        lines = stack_trace.split('\n')
        call_depth = 0

        for line in lines:
            if 'File "' in line and 'line' in line:
                call_depth += 1

        if call_depth > 20:
            issues.append("Very deep call stack - possible infinite recursion")

        # Check for repeated function calls
        function_calls = []
        for line in lines:
            if 'in ' in line and '(' in line:
                func_match = re.search(r'in (\w+)\(', line)
                if func_match:
                    function_calls.append(func_match.group(1))

        # Check for recursion
        if len(function_calls) != len(set(function_calls)):
            issues.append("Recursive function calls detected")

        return {"issues": issues, "call_depth": call_depth}

    def _analyze_error_location(self, file_path: str, line_number: int) -> Dict[str, Any]:
        """Analyze the code location where error occurred"""
        issues = []

        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                if line_number <= len(lines):
                    error_line = lines[line_number - 1].strip()

                    # Analyze the error line
                    if 'None' in error_line and ('+' in error_line or '-' in error_line):
                        issues.append("Possible None type operation")

                    if 'int(' in error_line and ('float' in error_line or 'str' in error_line):
                        issues.append("Potential type conversion issue")

                    # Check surrounding context
                    start_line = max(0, line_number - 3)
                    end_line = min(len(lines), line_number + 2)

                    context_lines = lines[start_line:end_line]
                    context = ''.join(context_lines)

                    if 'try:' in context and 'except' not in context:
                        issues.append("Incomplete try-except block")

        except Exception as e:
            issues.append(f"Could not analyze error location: {e}")

        return {"issues": issues}

    def _analyze_error_context(self, context: Dict[str, Any]) -> List[str]:
        """Analyze error context for contributing factors"""
        issues = []

        # Check system resources
        if context.get("memory_percent", 0) > 90:
            issues.append("High memory usage may have contributed to error")

        if context.get("cpu_percent", 0) > 95:
            issues.append("High CPU usage may indicate performance issues")

        # Check recent operations
        recent_ops = context.get("recent_operations", [])
        if len(recent_ops) > 10:
            issues.append("High frequency of operations may indicate stress")

        return issues

    def _generate_prevention_measures(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate prevention measures based on analysis"""
        measures = []

        primary_cause = analysis.get("primary_cause", "")

        if "recursion" in primary_cause:
            measures.extend([
                "Add recursion depth limits",
                "Implement iterative alternatives where possible",
                "Add cycle detection in data structures"
            ])

        elif "memory" in primary_cause:
            measures.extend([
                "Implement proper resource cleanup",
                "Use memory profiling tools",
                "Monitor memory usage in production"
            ])

        elif "type" in primary_cause:
            measures.extend([
                "Add comprehensive type checking",
                "Use type hints throughout codebase",
                "Implement input validation"
            ])

        # General measures
        measures.extend([
            "Add comprehensive error handling",
            "Implement logging for debugging",
            "Add input validation",
            "Use static type checking",
            "Implement unit tests",
            "Add performance monitoring"
        ])

        return list(set(measures))  # Remove duplicates


class Debugger:
    """Advanced debugging system with AI-powered analysis"""

    def __init__(self, healer):
        self.healer = healer
        self.jarvis = healer.jarvis
        self.logger = logging.getLogger('JARVIS.Debugger')

        # Analysis components
        self.root_cause_analyzer = RootCauseAnalyzer()
        self.call_graph_analyzer = CallGraphAnalyzer()
        self.variable_tracer = VariableTracer()

        # Debug sessions
        self.active_sessions: Dict[str, DebugSession] = {}
        self.session_history: List[DebugSession] = []

        # Configuration
        self.max_sessions = 10
        self.session_timeout = 3600  # 1 hour

        # Performance tracking
        self.analysis_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "average_analysis_time": 0.0,
            "error_patterns": {}
        }

    async def initialize(self):
        """Initialize the advanced debugger"""
        try:
            self.logger.info("Initializing advanced debugger system...")

            # Test analysis capabilities
            test_error = {
                "type": "test_error",
                "message": "Test error for initialization"
            }

            result = await self.debug_error(test_error)
            if result.get("success"):
                self.logger.info("Advanced debugger initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing debugger: {e}")
            raise

    async def debug_error(self, error: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform comprehensive error debugging and analysis"""
        start_time = time.time()
        session_id = f"debug_{int(time.time())}_{hash(str(error)) % 10000}"

        try:
            self.logger.info(f"Starting comprehensive error analysis: {session_id}")

            # Create debug session
            session = DebugSession(
                session_id=session_id,
                error_info=error,
                start_time=start_time
            )

            self.active_sessions[session_id] = session

            # Gather error context
            full_context = context or await self._gather_error_context(error)

            # Extract stack trace if available
            stack_trace = self._extract_stack_trace(error)
            session.stack_trace = stack_trace

            # Analyze root cause
            root_cause_analysis = self.root_cause_analyzer.analyze_root_cause(error, full_context)

            # Build call graph if code is available
            call_graph = {}
            if "file_path" in error and os.path.exists(error["file_path"]):
                try:
                    with open(error["file_path"], 'r') as f:
                        code = f.read()
                    call_graph = self.call_graph_analyzer.build_call_graph(code, error["file_path"])
                except Exception as e:
                    self.logger.debug(f"Could not build call graph: {e}")

            # Analyze execution path
            execution_analysis = {}
            if stack_trace:
                execution_analysis = self.call_graph_analyzer.analyze_execution_path(stack_trace)

            # Variable analysis
            variable_analysis = {}
            if session.variables:
                for var_name in session.variables.keys():
                    variable_analysis[var_name] = self.variable_tracer.analyze_variable_changes(var_name)

            # Detect anomalies
            anomalies = []
            for var_name in session.variables.keys():
                var_anomalies = self.variable_tracer.detect_anomalies(var_name)
                anomalies.extend(var_anomalies)

            # Generate recommendations
            recommendations = self._generate_debugging_recommendations(
                root_cause_analysis, execution_analysis, anomalies
            )

            # Calculate analysis confidence
            confidence = self._calculate_analysis_confidence(
                root_cause_analysis, execution_analysis, call_graph
            )

            analysis_time = time.time() - start_time

            # Update statistics
            self.analysis_stats["total_analyses"] += 1
            if root_cause_analysis.get("confidence", 0) > 0.5:
                self.analysis_stats["successful_analyses"] += 1

            # Update average time
            total_time = self.analysis_stats["average_analysis_time"] * (self.analysis_stats["total_analyses"] - 1)
            self.analysis_stats["average_analysis_time"] = (total_time + analysis_time) / self.analysis_stats["total_analyses"]

            # Store analysis results
            session.analysis_results = {
                "root_cause_analysis": root_cause_analysis,
                "execution_analysis": execution_analysis,
                "call_graph": call_graph,
                "variable_analysis": variable_analysis,
                "anomalies": anomalies,
                "recommendations": recommendations,
                "confidence": confidence,
                "analysis_time": analysis_time
            }

            # Move to history
            session.status = "completed"
            self.session_history.append(session)
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]

            # Cleanup old sessions
            await self._cleanup_old_sessions()

            result = {
                "success": True,
                "session_id": session_id,
                "root_cause": root_cause_analysis.get("primary_cause", "Unknown"),
                "confidence": confidence,
                "recommendations": recommendations,
                "debug_info": {
                    "stack_depth": len(stack_trace),
                    "variables_traced": len(variable_analysis),
                    "anomalies_detected": len(anomalies),
                    "call_graph_size": len(call_graph.get("functions", {}))
                },
                "analysis_time": analysis_time,
                "contributing_factors": root_cause_analysis.get("contributing_factors", []),
                "prevention_measures": root_cause_analysis.get("prevention_measures", [])
            }

            self.logger.info(f"Error analysis completed in {analysis_time:.2f}s with {confidence:.1f}% confidence")
            return result

        except Exception as e:
            self.logger.error(f"Error in comprehensive debugging: {e}")

            # Update failed session
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.status = "failed"
                session.analysis_results = {"error": str(e)}
                self.session_history.append(session)
                del self.active_sessions[session_id]

            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }

    async def _gather_error_context(self, error: Dict[str, Any]) -> Dict[str, Any]:
        """Gather comprehensive error context"""
        context = {
            "timestamp": time.time(),
            "system_info": {},
            "recent_operations": [],
            "memory_percent": 0,
            "cpu_percent": 0
        }

        try:
            # Get system information
            if hasattr(self.jarvis, 'system_core'):
                system_status = self.jarvis.system_core.get_system_status()
                context["system_info"] = system_status.get("system_info", {})
                resource_usage = system_status.get("resource_usage", {})
                context["memory_percent"] = resource_usage.get("memory_percent", 0)
                context["cpu_percent"] = resource_usage.get("cpu_percent", 0)

            # Get recent operations (simplified)
            context["recent_operations"] = ["operation_1", "operation_2"]  # Would be populated from actual logs

        except Exception as e:
            self.logger.debug(f"Could not gather full error context: {e}")

        return context

    def _extract_stack_trace(self, error: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and parse stack trace"""
        stack_trace = []

        traceback_str = error.get("traceback", "")
        if not traceback_str:
            return stack_trace

        lines = traceback_str.split('\n')
        current_frame = {}

        for line in lines:
            line = line.strip()
            if 'File "' in line and 'line' in line:
                # New frame
                if current_frame:
                    stack_trace.append(current_frame)

                current_frame = {}

                # Parse file and line
                file_match = re.search(r'File "([^"]+)", line (\d+)', line)
                if file_match:
                    current_frame["file"] = file_match.group(1)
                    current_frame["line"] = int(file_match.group(2))

            elif line.startswith('in ') and '(' in line:
                # Function name
                func_match = re.search(r'in (\w+)\(', line)
                if func_match:
                    current_frame["function"] = func_match.group(1)

        if current_frame:
            stack_trace.append(current_frame)

        return stack_trace

    def _generate_debugging_recommendations(self, root_cause: Dict[str, Any],
                                          execution: Dict[str, Any],
                                          anomalies: List[str]) -> List[str]:
        """Generate comprehensive debugging recommendations"""
        recommendations = []

        # Root cause based recommendations
        if root_cause.get("recommended_actions"):
            recommendations.extend(root_cause["recommended_actions"])

        # Execution path recommendations
        exec_issues = execution.get("potential_issues", [])
        recommendations.extend(exec_issues)

        # Anomaly-based recommendations
        for anomaly in anomalies:
            if "recursion" in anomaly.lower():
                recommendations.append("Add recursion depth checking")
            elif "memory" in anomaly.lower():
                recommendations.append("Implement memory profiling")
            elif "type" in anomaly.lower():
                recommendations.append("Add runtime type checking")

        # General recommendations
        recommendations.extend([
            "Add comprehensive logging",
            "Implement input validation",
            "Add error recovery mechanisms",
            "Create unit tests for error paths",
            "Use static analysis tools",
            "Implement monitoring and alerting"
        ])

        return list(set(recommendations))  # Remove duplicates

    def _calculate_analysis_confidence(self, root_cause: Dict[str, Any],
                                     execution: Dict[str, Any],
                                     call_graph: Dict[str, Any]) -> float:
        """Calculate overall analysis confidence"""
        confidence = 0.0
        factors = 0

        # Root cause confidence
        if "confidence" in root_cause:
            confidence += root_cause["confidence"] * 0.4
            factors += 0.4

        # Execution analysis quality
        if execution.get("call_chain"):
            chain_length = len(execution["call_chain"])
            confidence += min(1.0, chain_length / 10) * 0.3
            factors += 0.3

        # Call graph quality
        if call_graph.get("functions"):
            functions_count = len(call_graph["functions"])
            confidence += min(1.0, functions_count / 20) * 0.3
            factors += 0.3

        return confidence / factors if factors > 0 else 0.0

    async def _cleanup_old_sessions(self):
        """Clean up old debug sessions"""
        current_time = time.time()
        to_remove = []

        # Remove old active sessions
        for session_id, session in self.active_sessions.items():
            if current_time - session.start_time > self.session_timeout:
                to_remove.append(session_id)

        for session_id in to_remove:
            session = self.active_sessions[session_id]
            session.status = "timeout"
            self.session_history.append(session)
            del self.active_sessions[session_id]

        # Limit history size
        if len(self.session_history) > self.max_sessions:
            self.session_history = self.session_history[-self.max_sessions:]

    def get_debug_stats(self) -> Dict[str, Any]:
        """Get debugging statistics"""
        return {
            "total_analyses": self.analysis_stats["total_analyses"],
            "successful_analyses": self.analysis_stats["successful_analyses"],
            "success_rate": (self.analysis_stats["successful_analyses"] / max(1, self.analysis_stats["total_analyses"])),
            "average_analysis_time": self.analysis_stats["average_analysis_time"],
            "active_sessions": len(self.active_sessions),
            "completed_sessions": len(self.session_history)
        }

    def get_session_details(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get details of a specific debug session"""
        # Check active sessions
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            return {
                "session_id": session.session_id,
                "status": session.status,
                "start_time": session.start_time,
                "error_info": session.error_info,
                "stack_trace_length": len(session.stack_trace),
                "variables_count": len(session.variables)
            }

        # Check history
        for session in self.session_history:
            if session.session_id == session_id:
                return {
                    "session_id": session.session_id,
                    "status": session.status,
                    "start_time": session.start_time,
                    "end_time": getattr(session, 'end_time', None),
                    "error_info": session.error_info,
                    "analysis_results": session.analysis_results
                }

        return None

    async def shutdown(self):
        """Shutdown the advanced debugger"""
        try:
            self.logger.info("Shutting down advanced debugger...")

            # Complete any active sessions
            for session in self.active_sessions.values():
                session.status = "interrupted"
                self.session_history.append(session)

            self.active_sessions.clear()

            # Save debug statistics
            await self._save_debug_stats()

            self.logger.info("Advanced debugger shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down debugger: {e}")

    async def _save_debug_stats(self):
        """Save debugging statistics"""
        try:
            stats_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'debug_stats.json')

            os.makedirs(os.path.dirname(stats_file), exist_ok=True)

            with open(stats_file, 'w') as f:
                json.dump({
                    "stats": self.get_debug_stats(),
                    "last_updated": time.time()
                }, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving debug stats: {e}")