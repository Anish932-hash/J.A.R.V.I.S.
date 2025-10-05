"""
J.A.R.V.I.S. Advanced Validator
Comprehensive code validation, security scanning, and quality assessment system
"""

import os
import ast
import time
import json
import logging
import asyncio
import inspect
import builtins
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor


@dataclass
class ValidationResult:
    """Represents a validation result"""
    validator_name: str
    category: str
    severity: str  # critical, high, medium, low, info
    message: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    code_snippet: Optional[str] = None
    suggestion: Optional[str] = None
    rule_id: Optional[str] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    code_hash: str
    timestamp: float
    overall_score: float
    security_score: float
    performance_score: float
    maintainability_score: float
    reliability_score: float
    results: List[ValidationResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


class SecurityValidator:
    """Advanced security validation"""

    def __init__(self):
        self.security_patterns = {
            "dangerous_functions": {
                "eval": {"severity": "critical", "message": "Use of eval() is extremely dangerous"},
                "exec": {"severity": "critical", "message": "Use of exec() can execute arbitrary code"},
                "input": {"severity": "high", "message": "Use of input() can lead to code injection"},
                "open": {"severity": "medium", "message": "File operations should be validated"},
                "subprocess.call": {"severity": "high", "message": "Subprocess calls can be dangerous"},
                "os.system": {"severity": "high", "message": "System commands should be avoided"},
                "pickle.load": {"severity": "high", "message": "Pickle can execute arbitrary code"},
                "__import__": {"severity": "medium", "message": "Dynamic imports can be risky"}
            },
            "weak_crypto": {
                "md5": {"severity": "medium", "message": "MD5 is cryptographically weak"},
                "sha1": {"severity": "medium", "message": "SHA1 is cryptographically weak"},
                "random": {"severity": "low", "message": "Use secrets module for cryptographic randomness"}
            },
            "information_disclosure": [
                r"password\s*=\s*['\"][^'\"]*['\"]",
                r"key\s*=\s*['\"][^'\"]*['\"]",
                r"token\s*=\s*['\"][^'\"]*['\"]",
                r"secret\s*=\s*['\"][^'\"]*['\"]"
            ]
        }

    def validate(self, code: str, tree: ast.AST) -> List[ValidationResult]:
        """Perform security validation"""
        results = []

        # Check for dangerous function calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node)
                if func_name in self.security_patterns["dangerous_functions"]:
                    pattern = self.security_patterns["dangerous_functions"][func_name]
                    results.append(ValidationResult(
                        validator_name="security",
                        category="dangerous_functions",
                        severity=pattern["severity"],
                        message=pattern["message"],
                        line_number=getattr(node, 'lineno', None),
                        rule_id=f"SEC_{func_name.upper()}"
                    ))

        # Check for weak cryptography
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                attr_name = self._get_attribute_name(node)
                if attr_name in self.security_patterns["weak_crypto"]:
                    pattern = self.security_patterns["weak_crypto"][attr_name]
                    results.append(ValidationResult(
                        validator_name="security",
                        category="weak_crypto",
                        severity=pattern["severity"],
                        message=pattern["message"],
                        line_number=getattr(node, 'lineno', None),
                        rule_id=f"SEC_WEAK_{attr_name.upper()}"
                    ))

        # Check for hardcoded secrets
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            for pattern in self.security_patterns["information_disclosure"]:
                if re.search(pattern, line, re.IGNORECASE):
                    results.append(ValidationResult(
                        validator_name="security",
                        category="information_disclosure",
                        severity="high",
                        message="Potential hardcoded sensitive information",
                        line_number=i,
                        code_snippet=line.strip(),
                        rule_id="SEC_HARDCODED_SECRET"
                    ))

        return results

    def _get_function_name(self, node: ast.Call) -> str:
        """Get function name from call node"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return f"{self._get_attribute_name(node.func)}"
        return ""

    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get full attribute name"""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._get_attribute_name(node.value)}.{node.attr}"
        return node.attr


class PerformanceValidator:
    """Performance validation and optimization suggestions"""

    def __init__(self):
        self.performance_patterns = {
            "inefficient_loops": [],
            "memory_issues": [],
            "blocking_operations": [
                "time.sleep", "input", "open", "requests.get", "urllib.request.urlopen"
            ]
        }

    def validate(self, code: str, tree: ast.AST) -> List[ValidationResult]:
        """Perform performance validation"""
        results = []

        # Check for inefficient patterns
        for node in ast.walk(tree):
            # Check for loops with expensive operations
            if isinstance(node, ast.For):
                loop_body = self._get_loop_body_complexity(node)
                if loop_body > 10:  # Arbitrary complexity threshold
                    results.append(ValidationResult(
                        validator_name="performance",
                        category="loop_efficiency",
                        severity="medium",
                        message="Complex operations inside loop may impact performance",
                        line_number=getattr(node, 'lineno', None),
                        suggestion="Consider optimizing loop operations or using list comprehensions"
                    ))

            # Check for blocking operations
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node)
                if func_name in self.performance_patterns["blocking_operations"]:
                    results.append(ValidationResult(
                        validator_name="performance",
                        category="blocking_operations",
                        severity="low",
                        message=f"Blocking operation '{func_name}' may cause performance issues",
                        line_number=getattr(node, 'lineno', None),
                        suggestion="Consider using async alternatives"
                    ))

        # Check for potential memory issues
        memory_issues = self._check_memory_usage(code)
        results.extend(memory_issues)

        return results

    def _get_loop_body_complexity(self, loop_node: ast.For) -> int:
        """Calculate complexity of loop body"""
        complexity = 0
        for node in ast.walk(loop_node):
            if isinstance(node, (ast.Call, ast.BinOp, ast.Compare)):
                complexity += 1
        return complexity

    def _get_function_name(self, node: ast.Call) -> str:
        """Get function name from call node"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
        return ""

    def _check_memory_usage(self, code: str) -> List[ValidationResult]:
        """Check for potential memory issues"""
        results = []

        # Check for large data structures
        if "list(" in code and "range(" in code:
            results.append(ValidationResult(
                validator_name="performance",
                category="memory_usage",
                severity="medium",
                message="Creating large lists with range() may consume excessive memory",
                suggestion="Consider using generators or itertools for large sequences"
            ))

        # Check for string concatenation in loops
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            if '+' in line and ('for ' in line or 'while ' in line):
                results.append(ValidationResult(
                    validator_name="performance",
                    category="string_concatenation",
                    severity="low",
                    message="String concatenation in loop may be inefficient",
                    line_number=i,
                    suggestion="Use ''.join() or f-strings for better performance"
                ))

        return results


class MaintainabilityValidator:
    """Code maintainability and readability validation"""

    def __init__(self):
        self.max_line_length = 100
        self.max_function_length = 50
        self.max_class_length = 300

    def validate(self, code: str, tree: ast.AST) -> List[ValidationResult]:
        """Perform maintainability validation"""
        results = []

        lines = code.split('\n')

        # Check line lengths
        for i, line in enumerate(lines, 1):
            if len(line) > self.max_line_length:
                results.append(ValidationResult(
                    validator_name="maintainability",
                    category="line_length",
                    severity="low",
                    message=f"Line too long ({len(line)} characters, max {self.max_line_length})",
                    line_number=i,
                    suggestion="Break long lines for better readability"
                ))

        # Check function lengths
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_length = self._get_node_length(node, lines)
                if func_length > self.max_function_length:
                    results.append(ValidationResult(
                        validator_name="maintainability",
                        category="function_length",
                        severity="medium",
                        message=f"Function '{node.name}' is too long ({func_length} lines)",
                        line_number=node.lineno,
                        suggestion="Consider breaking down into smaller functions"
                    ))

        # Check class lengths
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_length = self._get_node_length(node, lines)
                if class_length > self.max_class_length:
                    results.append(ValidationResult(
                        validator_name="maintainability",
                        category="class_length",
                        severity="medium",
                        message=f"Class '{node.name}' is too long ({class_length} lines)",
                        line_number=node.lineno,
                        suggestion="Consider splitting into multiple classes"
                    ))

        # Check for missing docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not self._has_docstring(node):
                    results.append(ValidationResult(
                        validator_name="maintainability",
                        category="documentation",
                        severity="low",
                        message=f"Missing docstring for {type(node).__name__.lower()} '{node.name}'",
                        line_number=node.lineno,
                        suggestion="Add docstring to document functionality"
                    ))

        # Check naming conventions
        naming_issues = self._check_naming_conventions(tree)
        results.extend(naming_issues)

        return results

    def _get_node_length(self, node: ast.AST, lines: List[str]) -> int:
        """Get the length of a node in lines"""
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            return node.end_lineno - node.lineno + 1
        return 0

    def _has_docstring(self, node: ast.AST) -> bool:
        """Check if node has a docstring"""
        if not hasattr(node, 'body') or not node.body:
            return False

        first_stmt = node.body[0]
        if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Str):
            return True

        return False

    def _check_naming_conventions(self, tree: ast.AST) -> List[ValidationResult]:
        """Check Python naming conventions"""
        results = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                    results.append(ValidationResult(
                        validator_name="maintainability",
                        category="naming",
                        severity="low",
                        message=f"Function name '{node.name}' doesn't follow snake_case convention",
                        line_number=node.lineno,
                        suggestion="Use snake_case for function names"
                    ))

            elif isinstance(node, ast.ClassDef):
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    results.append(ValidationResult(
                        validator_name="maintainability",
                        category="naming",
                        severity="low",
                        message=f"Class name '{node.name}' doesn't follow PascalCase convention",
                        line_number=node.lineno,
                        suggestion="Use PascalCase for class names"
                    ))

            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                # Variable assignment
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.id):
                    results.append(ValidationResult(
                        validator_name="maintainability",
                        category="naming",
                        severity="low",
                        message=f"Variable name '{node.id}' doesn't follow snake_case convention",
                        line_number=getattr(node, 'lineno', None),
                        suggestion="Use snake_case for variable names"
                    ))

        return results


class ReliabilityValidator:
    """Code reliability and error handling validation"""

    def __init__(self):
        pass

    def validate(self, code: str, tree: ast.AST) -> List[ValidationResult]:
        """Perform reliability validation"""
        results = []

        # Check for error handling
        has_try_except = False
        has_logging = False

        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                has_try_except = True

            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node)
                if 'log' in func_name.lower() or 'logger' in func_name.lower():
                    has_logging = True

        # Check for bare except clauses
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:  # Bare except
                    results.append(ValidationResult(
                        validator_name="reliability",
                        category="error_handling",
                        severity="medium",
                        message="Bare 'except:' clause catches all exceptions",
                        line_number=getattr(node, 'lineno', None),
                        suggestion="Specify exception types to catch"
                    ))

        # Check for missing error handling in risky operations
        risky_operations = ['open', 'connect', 'get', 'post', 'load', 'dump']
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node)
                if any(op in func_name for op in risky_operations):
                    # Check if this call is inside a try block
                    if not self._is_inside_try(node, tree):
                        results.append(ValidationResult(
                            validator_name="reliability",
                            category="error_handling",
                            severity="low",
                            message=f"Risky operation '{func_name}' not wrapped in try-except",
                            line_number=getattr(node, 'lineno', None),
                            suggestion="Add error handling for this operation"
                        ))

        # Overall reliability assessment
        if not has_try_except:
            results.append(ValidationResult(
                validator_name="reliability",
                category="error_handling",
                severity="info",
                message="No try-except blocks found in code",
                suggestion="Consider adding error handling where appropriate"
            ))

        if not has_logging:
            results.append(ValidationResult(
                validator_name="reliability",
                category="logging",
                severity="info",
                message="No logging statements found",
                suggestion="Add logging for better debugging and monitoring"
            ))

        return results

    def _get_function_name(self, node: ast.Call) -> str:
        """Get function name from call node"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
        return ""

    def _is_inside_try(self, target_node: ast.AST, tree: ast.AST) -> bool:
        """Check if a node is inside a try block"""
        # This is a simplified check - would need more sophisticated AST analysis
        return False


class CodeValidator:
    """Comprehensive code validation engine"""

    def __init__(self):
        self.validators = {
            "security": SecurityValidator(),
            "performance": PerformanceValidator(),
            "maintainability": MaintainabilityValidator(),
            "reliability": ReliabilityValidator()
        }

    def validate_code(self, code: str) -> ValidationReport:
        """Perform comprehensive code validation"""
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        timestamp = time.time()

        try:
            # Parse AST
            tree = ast.parse(code)

            # Run all validators
            all_results = []
            for validator_name, validator in self.validators.items():
                results = validator.validate(code, tree)
                all_results.extend(results)

            # Calculate scores
            scores = self._calculate_scores(all_results)

            # Create summary
            summary = self._create_summary(all_results)

            return ValidationReport(
                code_hash=code_hash,
                timestamp=timestamp,
                overall_score=scores["overall"],
                security_score=scores["security"],
                performance_score=scores["performance"],
                maintainability_score=scores["maintainability"],
                reliability_score=scores["reliability"],
                results=all_results,
                summary=summary
            )

        except SyntaxError as e:
            # Handle syntax errors
            error_result = ValidationResult(
                validator_name="syntax",
                category="syntax_error",
                severity="critical",
                message=f"Syntax error: {e}",
                line_number=e.lineno,
                suggestion="Fix syntax error before validation"
            )

            return ValidationReport(
                code_hash=code_hash,
                timestamp=timestamp,
                overall_score=0.0,
                security_score=0.0,
                performance_score=0.0,
                maintainability_score=0.0,
                reliability_score=0.0,
                results=[error_result],
                summary={"syntax_errors": 1, "total_issues": 1}
            )

    def _calculate_scores(self, results: List[ValidationResult]) -> Dict[str, float]:
        """Calculate validation scores"""
        # Group results by category
        category_counts = {}
        severity_weights = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4,
            "info": 0.2
        }

        for result in results:
            category = result.category
            severity = result.severity

            if category not in category_counts:
                category_counts[category] = {"count": 0, "weighted_score": 0.0}

            category_counts[category]["count"] += 1
            category_counts[category]["weighted_score"] += severity_weights.get(severity, 0.5)

        # Calculate category scores (lower issues = higher score)
        scores = {}
        for category in ["security", "performance", "maintainability", "reliability"]:
            if category in category_counts:
                issue_score = category_counts[category]["weighted_score"]
                # Convert to 0-100 scale (more issues = lower score)
                scores[category] = max(0, 100 - (issue_score * 20))
            else:
                scores[category] = 100.0  # No issues = perfect score

        # Overall score is average of category scores
        scores["overall"] = sum(scores.values()) / len(scores)

        return scores

    def _create_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Create validation summary"""
        summary = {
            "total_issues": len(results),
            "issues_by_severity": {},
            "issues_by_category": {},
            "issues_by_validator": {}
        }

        for result in results:
            # Count by severity
            severity = result.severity
            summary["issues_by_severity"][severity] = summary["issues_by_severity"].get(severity, 0) + 1

            # Count by category
            category = result.category
            summary["issues_by_category"][category] = summary["issues_by_category"].get(category, 0) + 1

            # Count by validator
            validator = result.validator_name
            summary["issues_by_validator"][validator] = summary["issues_by_validator"].get(validator, 0) + 1

        return summary


class Validator:
    """Advanced code validation and security scanning system"""

    def __init__(self, development_engine):
        self.engine = development_engine
        self.jarvis = development_engine.jarvis
        self.logger = logging.getLogger('JARVIS.Validator')

        # Core validation components
        self.code_validator = CodeValidator()

        # Validation history and caching
        self.validation_cache: Dict[str, ValidationReport] = {}
        self.validation_history: List[ValidationReport] = []
        self.cache_ttl = 3600  # 1 hour

        # Configuration
        self.strict_mode = False
        self.max_history_size = 1000

    async def initialize(self):
        """Initialize the advanced validator"""
        try:
            self.logger.info("Initializing advanced validator...")

            # Load validation history
            await self._load_validation_history()

            self.logger.info("✓ Advanced validator initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing validator: {e}")
            raise

    async def validate_code(self, code: str, use_cache: bool = True) -> Dict[str, Any]:
        """Perform comprehensive code validation"""
        try:
            code_hash = hashlib.sha256(code.encode()).hexdigest()

            # Check cache
            if use_cache and code_hash in self.validation_cache:
                cached_report = self.validation_cache[code_hash]
                if time.time() - cached_report.timestamp < self.cache_ttl:
                    self.logger.debug("Using cached validation results")
                    return self._report_to_dict(cached_report)

            # Perform validation
            self.logger.debug("Performing code validation...")
            report = await asyncio.get_event_loop().run_in_executor(
                None, self.code_validator.validate_code, code
            )

            # Cache result
            self.validation_cache[code_hash] = report

            # Add to history
            self.validation_history.append(report)

            # Maintain history size
            if len(self.validation_history) > self.max_history_size:
                self.validation_history.pop(0)

            # Convert to dict for return
            result = self._report_to_dict(report)

            # Log critical issues
            critical_issues = [r for r in report.results if r.severity == "critical"]
            if critical_issues:
                self.logger.warning(f"Critical validation issues found: {len(critical_issues)}")

            return result

        except Exception as e:
            self.logger.error(f"Error validating code: {e}")
            return {
                "success": False,
                "error": str(e),
                "overall_score": 0,
                "security_score": 0,
                "performance_score": 0,
                "maintainability_score": 0,
                "reliability_score": 0,
                "issues": [],
                "summary": {}
            }

    def _report_to_dict(self, report: ValidationReport) -> Dict[str, Any]:
        """Convert validation report to dictionary"""
        return {
            "success": True,
            "code_hash": report.code_hash,
            "timestamp": report.timestamp,
            "overall_score": round(report.overall_score, 2),
            "security_score": round(report.security_score, 2),
            "performance_score": round(report.performance_score, 2),
            "maintainability_score": round(report.maintainability_score, 2),
            "reliability_score": round(report.reliability_score, 2),
            "issues": [
                {
                    "validator": r.validator_name,
                    "category": r.category,
                    "severity": r.severity,
                    "message": r.message,
                    "line_number": r.line_number,
                    "suggestion": r.suggestion,
                    "rule_id": r.rule_id
                }
                for r in report.results
            ],
            "summary": report.summary
        }

    async def validate_file(self, file_path: str) -> Dict[str, Any]:
        """Validate a file"""
        try:
            if not os.path.exists(file_path):
                return {"success": False, "error": f"File not found: {file_path}"}

            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            return await self.validate_code(code)

        except Exception as e:
            self.logger.error(f"Error validating file {file_path}: {e}")
            return {"success": False, "error": str(e)}

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        if not self.validation_history:
            return {"total_validations": 0}

        total_validations = len(self.validation_history)
        avg_overall_score = sum(r.overall_score for r in self.validation_history) / total_validations
        avg_security_score = sum(r.security_score for r in self.validation_history) / total_validations

        # Count issues by severity
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for report in self.validation_history:
            for result in report.results:
                severity_counts[result.severity] = severity_counts.get(result.severity, 0) + 1

        return {
            "total_validations": total_validations,
            "average_overall_score": round(avg_overall_score, 2),
            "average_security_score": round(avg_security_score, 2),
            "issues_by_severity": severity_counts,
            "cache_size": len(self.validation_cache),
            "cache_hit_rate": 0  # Would need to track cache hits separately
        }

    def clear_cache(self):
        """Clear validation cache"""
        self.validation_cache.clear()
        self.logger.info("Validation cache cleared")

    def set_strict_mode(self, enabled: bool):
        """Enable or disable strict validation mode"""
        self.strict_mode = enabled
        self.logger.info(f"Strict mode {'enabled' if enabled else 'disabled'}")

    async def _load_validation_history(self):
        """Load validation history from storage"""
        try:
            history_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'validation_history.json')
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    # Would need to reconstruct ValidationReport objects
                    self.logger.debug("Loaded validation history")

        except Exception as e:
            self.logger.debug(f"Could not load validation history: {e}")

    async def shutdown(self):
        """Shutdown the advanced validator"""
        try:
            self.logger.info("Shutting down advanced validator...")

            # Save validation history
            await self._save_validation_history()

            # Clear cache
            self.clear_cache()

            self.logger.info("Advanced validator shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down validator: {e}")

    async def _save_validation_history(self):
        """Save validation history"""
        try:
            history_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'validation_history.json')

            os.makedirs(os.path.dirname(history_file), exist_ok=True)

            # Convert reports to serializable format
            history_data = []
            for report in self.validation_history[-500:]:  # Last 500 validations
                history_data.append({
                    "code_hash": report.code_hash,
                    "timestamp": report.timestamp,
                    "overall_score": report.overall_score,
                    "security_score": report.security_score,
                    "performance_score": report.performance_score,
                    "maintainability_score": report.maintainability_score,
                    "reliability_score": report.reliability_score,
                    "issues_count": len(report.results),
                    "summary": report.summary
                })

            data = {
                "history": history_data,
                "stats": self.get_validation_stats(),
                "last_updated": time.time()
            }

            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Error saving validation history: {e}")