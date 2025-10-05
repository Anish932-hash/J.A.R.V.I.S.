"""
J.A.R.V.I.S. Advanced Fix Generator
AI-powered comprehensive fix generation with code analysis and automated repair
"""

import os
import ast
import time
import json
import logging
import difflib
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor


@dataclass
class FixCandidate:
    """Represents a potential fix"""
    description: str
    code_changes: List[Dict[str, Any]]
    confidence: float
    risk_level: str
    test_cases: List[str] = field(default_factory=list)
    rollback_plan: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodePatch:
    """Represents a code patch"""
    file_path: str
    line_number: int
    old_code: str
    new_code: str
    patch_type: str
    context: Dict[str, Any] = field(default_factory=dict)


class CodeAnalyzer:
    """Analyzes code for fix generation"""

    def __init__(self):
        self.patterns = {
            "null_pointer": {
                "pattern": r"\bNone\s*[\+\-\*\/]",
                "fix_template": "if {var} is not None: {operation}",
                "description": "Add None check before operation"
            },
            "type_conversion": {
                "pattern": r"(int|float|str)\([^)]*\)",
                "fix_template": "try: {conversion} except (ValueError, TypeError): {fallback}",
                "description": "Add error handling for type conversion"
            },
            "key_access": {
                "pattern": r"\w+\[\w+\]",
                "fix_template": "{dict}.get({key}, {default})",
                "description": "Use dict.get() for safe key access"
            },
            "attribute_access": {
                "pattern": r"\w+\.\w+",
                "fix_template": "getattr({obj}, '{attr}', {default})",
                "description": "Use getattr() for safe attribute access"
            },
            "division_by_zero": {
                "pattern": r"/\s*\w+",
                "fix_template": "if {divisor} != 0: {operation}",
                "description": "Add division by zero check"
            }
        }

    def analyze_code_for_fixes(self, code: str, error_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze code and suggest fixes"""
        fixes = []

        try:
            tree = ast.parse(code)

            # Analyze AST for common issues
            issues = self._analyze_ast_issues(tree)

            for issue in issues:
                fix = self._generate_fix_for_issue(issue, code)
                if fix:
                    fixes.append(fix)

            # Pattern-based fixes
            pattern_fixes = self._apply_pattern_fixes(code, error_info)
            fixes.extend(pattern_fixes)

        except SyntaxError:
            # For syntax errors, suggest basic fixes
            fixes.append({
                "type": "syntax_fix",
                "description": "Fix syntax error",
                "code_changes": [],
                "confidence": 0.3
            })

        return fixes

    def _analyze_ast_issues(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze AST for potential issues"""
        issues = []

        class IssueVisitor(ast.NodeVisitor):
            def __init__(self):
                self.issues = []

            def visit_BinOp(self, node):
                # Check for potential None operations
                if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                    if self._is_none_check_needed(node.left) or self._is_none_check_needed(node.right):
                        self.issues.append({
                            "type": "potential_none_operation",
                            "node": node,
                            "line": node.lineno
                        })
                self.generic_visit(node)

            def visit_Subscript(self, node):
                # Check for dictionary access without .get()
                if isinstance(node.value, ast.Name):
                    self.issues.append({
                        "type": "unsafe_dict_access",
                        "node": node,
                        "line": node.lineno
                    })
                self.generic_visit(node)

            def visit_Attribute(self, node):
                # Check for attribute access that might fail
                if isinstance(node.value, ast.Name):
                    self.issues.append({
                        "type": "unsafe_attribute_access",
                        "node": node,
                        "line": node.lineno
                    })
                self.generic_visit(node)

            def _is_none_check_needed(self, node):
                """Check if None check might be needed"""
                return isinstance(node, ast.Name) or isinstance(node, ast.Attribute)

        visitor = IssueVisitor()
        visitor.visit(tree)

        return visitor.issues

    def _generate_fix_for_issue(self, issue: Dict[str, Any], code: str) -> Optional[Dict[str, Any]]:
        """Generate a fix for a specific issue"""
        issue_type = issue.get("type")
        line_number = issue.get("line", 1)

        if issue_type == "potential_none_operation":
            return {
                "type": "none_check",
                "description": "Add None check for operation",
                "line_number": line_number,
                "fix_type": "wrap_with_check",
                "confidence": 0.7
            }

        elif issue_type == "unsafe_dict_access":
            return {
                "type": "safe_dict_access",
                "description": "Use dict.get() for safe access",
                "line_number": line_number,
                "fix_type": "replace_operation",
                "confidence": 0.8
            }

        elif issue_type == "unsafe_attribute_access":
            return {
                "type": "safe_attribute_access",
                "description": "Use getattr() for safe access",
                "line_number": line_number,
                "fix_type": "replace_operation",
                "confidence": 0.6
            }

        return None

    def _apply_pattern_fixes(self, code: str, error_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply pattern-based fixes"""
        fixes = []
        lines = code.split('\n')

        for i, line in enumerate(lines, 1):
            for pattern_name, pattern_data in self.patterns.items():
                if re.search(pattern_data["pattern"], line):
                    fixes.append({
                        "type": pattern_name,
                        "description": pattern_data["description"],
                        "line_number": i,
                        "pattern": pattern_name,
                        "fix_type": "pattern_replace",
                        "confidence": 0.5
                    })

        return fixes


class FixValidator:
    """Validates generated fixes"""

    def __init__(self):
        pass

    def validate_fix(self, original_code: str, fix: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a fix before applying"""
        validation = {
            "is_valid": True,
            "warnings": [],
            "risk_assessment": "low",
            "test_recommendations": []
        }

        # Check if fix changes too much code
        code_changes = fix.get("code_changes", [])
        total_lines_changed = sum(len(change.get("new_code", "").split('\n')) for change in code_changes)

        if total_lines_changed > 10:
            validation["warnings"].append("Fix changes many lines - high risk")
            validation["risk_assessment"] = "high"

        # Check for potential infinite loops
        for change in code_changes:
            new_code = change.get("new_code", "")
            if "while True:" in new_code and "break" not in new_code:
                validation["warnings"].append("Potential infinite loop introduced")
                validation["is_valid"] = False

        # Check for syntax validity
        try:
            # Apply fix to test syntax
            test_code = self._apply_fix_to_code(original_code, fix)
            ast.parse(test_code)
        except SyntaxError as e:
            validation["warnings"].append(f"Syntax error in fix: {e}")
            validation["is_valid"] = False

        # Generate test recommendations
        if fix.get("type") == "none_check":
            validation["test_recommendations"].append("Test with None input")
        elif "dict" in fix.get("type", ""):
            validation["test_recommendations"].append("Test with missing keys")
        elif "attribute" in fix.get("type", ""):
            validation["test_recommendations"].append("Test with missing attributes")

        return validation

    def _apply_fix_to_code(self, code: str, fix: Dict[str, Any]) -> str:
        """Apply fix to code for testing"""
        # Simplified implementation - would need full patch application logic
        return code


class FixApplier:
    """Applies fixes to code"""

    def __init__(self):
        pass

    def apply_fix(self, file_path: str, fix: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a fix to a file"""
        try:
            # Read original file
            with open(file_path, 'r') as f:
                original_code = f.read()

            # Apply the fix
            modified_code = self._apply_fix_changes(original_code, fix)

            # Create backup
            backup_path = f"{file_path}.backup.{int(time.time())}"
            with open(backup_path, 'w') as f:
                f.write(original_code)

            # Write modified code
            with open(file_path, 'w') as f:
                f.write(modified_code)

            return {
                "success": True,
                "backup_path": backup_path,
                "changes_applied": len(fix.get("code_changes", [])),
                "diff": self._generate_diff(original_code, modified_code)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _apply_fix_changes(self, code: str, fix: Dict[str, Any]) -> str:
        """Apply fix changes to code"""
        lines = code.split('\n')
        code_changes = fix.get("code_changes", [])

        # Sort changes by line number (reverse order to avoid offset issues)
        code_changes.sort(key=lambda x: x.get("line_number", 0), reverse=True)

        for change in code_changes:
            line_number = change.get("line_number", 1) - 1  # Convert to 0-based
            old_code = change.get("old_code", "")
            new_code = change.get("new_code", "")

            if line_number < len(lines):
                # Replace the line
                lines[line_number] = new_code

        return '\n'.join(lines)

    def _generate_diff(self, original: str, modified: str) -> str:
        """Generate diff between original and modified code"""
        diff = list(difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile='original',
            tofile='modified',
            lineterm=''
        ))
        return ''.join(diff)

    def rollback_fix(self, file_path: str, backup_path: str) -> bool:
        """Rollback a fix using backup"""
        try:
            with open(backup_path, 'r') as f:
                backup_code = f.read()

            with open(file_path, 'w') as f:
                f.write(backup_code)

            # Remove backup file
            os.remove(backup_path)

            return True

        except Exception as e:
            logging.error(f"Error rolling back fix: {e}")
            return False


class FixGenerator:
    """Advanced AI-powered fix generation system"""

    def __init__(self, healer):
        self.healer = healer
        self.jarvis = healer.jarvis
        self.logger = logging.getLogger('JARVIS.FixGenerator')

        # Fix generation components
        self.code_analyzer = CodeAnalyzer()
        self.fix_validator = FixValidator()
        self.fix_applier = FixApplier()

        # Fix database
        self.fix_history = []
        self.successful_fixes = {}
        self.failed_fixes = {}

        # Configuration
        self.max_fix_candidates = 5
        self.min_confidence_threshold = 0.3

    async def initialize(self):
        """Initialize the advanced fix generator"""
        try:
            self.logger.info("Initializing advanced fix generator...")

            # Load fix patterns and history
            await self._load_fix_database()

            # Test fix generation
            test_error = {"type": "test_error", "message": "x + None"}
            result = await self.generate_fix(test_error, {})
            if result.get("success"):
                self.logger.info("Advanced fix generator initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing fix generator: {e}")
            raise

    async def generate_fix(self, error: Dict[str, Any], debug_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive fix for error"""
        start_time = time.time()

        try:
            self.logger.info(f"Generating fix for error: {error.get('type', 'unknown')}")

            # Analyze error and gather context
            error_analysis = await self._analyze_error(error, debug_info)

            # Generate fix candidates
            fix_candidates = await self._generate_fix_candidates(error_analysis)

            # Validate and rank candidates
            validated_candidates = await self._validate_fix_candidates(fix_candidates, error_analysis)

            # Select best fix
            best_fix = self._select_best_fix(validated_candidates)

            if not best_fix:
                return {
                    "success": False,
                    "error": "No suitable fix found",
                    "analysis": error_analysis
                }

            # Generate implementation details
            implementation = await self._generate_fix_implementation(best_fix, error_analysis)

            generation_time = time.time() - start_time

            result = {
                "success": True,
                "fix": implementation,
                "confidence": best_fix.get("confidence", 0),
                "risk_level": best_fix.get("risk_level", "medium"),
                "analysis": error_analysis,
                "alternatives": len(validated_candidates) - 1,
                "generation_time": generation_time,
                "test_cases": best_fix.get("test_cases", []),
                "rollback_plan": best_fix.get("rollback_plan", {})
            }

            # Record fix attempt
            self.fix_history.append({
                "timestamp": time.time(),
                "error_type": error.get("type"),
                "fix_type": implementation.get("type"),
                "confidence": result["confidence"],
                "success": None  # Will be updated when fix is applied
            })

            self.logger.info(f"Fix generated in {generation_time:.2f}s with {result['confidence']:.1f}% confidence")
            return result

        except Exception as e:
            self.logger.error(f"Error generating fix: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _analyze_error(self, error: Dict[str, Any], debug_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error for fix generation"""
        analysis = {
            "error_type": error.get("type", "unknown"),
            "error_message": error.get("message", ""),
            "file_path": error.get("file_path"),
            "line_number": error.get("line_number"),
            "code_context": "",
            "variables_involved": [],
            "error_category": "unknown",
            "severity": "medium"
        }

        # Categorize error
        error_message = analysis["error_message"].lower()

        if "attributeerror" in error_message or "has no attribute" in error_message:
            analysis["error_category"] = "attribute_error"
        elif "keyerror" in error_message or "key not found" in error_message:
            analysis["error_category"] = "key_error"
        elif "typeerror" in error_message or "unsupported operand" in error_message:
            analysis["error_category"] = "type_error"
        elif "importerror" in error_message or "module not found" in error_message:
            analysis["error_category"] = "import_error"
        elif "valueerror" in error_message or "invalid" in error_message:
            analysis["error_category"] = "value_error"
        elif "ioerror" in error_message or "file not found" in error_message:
            analysis["error_category"] = "io_error"

        # Get code context
        if analysis["file_path"] and analysis["line_number"]:
            analysis["code_context"] = await self._get_code_context(
                analysis["file_path"], analysis["line_number"]
            )

        # Extract variables from error message
        analysis["variables_involved"] = self._extract_variables_from_error(error_message)

        # Assess severity
        if "critical" in error_message or "fatal" in error_message:
            analysis["severity"] = "critical"
        elif "high" in error_message or "severe" in error_message:
            analysis["severity"] = "high"
        elif "low" in error_message or "minor" in error_message:
            analysis["severity"] = "low"

        return analysis

    async def _get_code_context(self, file_path: str, line_number: int, context_lines: int = 3) -> str:
        """Get code context around error line"""
        try:
            if not os.path.exists(file_path):
                return ""

            with open(file_path, 'r') as f:
                lines = f.readlines()

            start_line = max(0, line_number - context_lines - 1)
            end_line = min(len(lines), line_number + context_lines)

            context_lines = lines[start_line:end_line]
            return ''.join(context_lines)

        except Exception as e:
            self.logger.debug(f"Could not get code context: {e}")
            return ""

    def _extract_variables_from_error(self, error_message: str) -> List[str]:
        """Extract variable names from error message"""
        variables = []

        # Look for quoted strings that might be variable names
        import re
        quoted_vars = re.findall(r"'([^']+)'", error_message)
        variables.extend(quoted_vars)

        # Look for variable-like patterns
        var_patterns = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', error_message)
        # Filter out common words
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'an', 'a'}
        variables.extend([var for var in var_patterns if var not in common_words and len(var) > 1])

        return list(set(variables))[:5]  # Limit to 5 most relevant

    async def _generate_fix_candidates(self, error_analysis: Dict[str, Any]) -> List[FixCandidate]:
        """Generate candidate fixes for the error"""
        candidates = []

        error_category = error_analysis.get("error_category")
        code_context = error_analysis.get("code_context", "")

        # Generate fixes based on error category
        if error_category == "attribute_error":
            candidates.extend(self._generate_attribute_error_fixes(error_analysis))
        elif error_category == "key_error":
            candidates.extend(self._generate_key_error_fixes(error_analysis))
        elif error_category == "type_error":
            candidates.extend(self._generate_type_error_fixes(error_analysis))
        elif error_category == "import_error":
            candidates.extend(self._generate_import_error_fixes(error_analysis))
        elif error_category == "value_error":
            candidates.extend(self._generate_value_error_fixes(error_analysis))
        elif error_category == "io_error":
            candidates.extend(self._generate_io_error_fixes(error_analysis))

        # Generate generic fixes
        candidates.extend(self._generate_generic_fixes(error_analysis))

        # Analyze code for additional fixes
        if code_context:
            code_fixes = self.code_analyzer.analyze_code_for_fixes(code_context, error_analysis)
            for fix_data in code_fixes:
                candidate = FixCandidate(
                    description=fix_data.get("description", "Code analysis fix"),
                    code_changes=[],  # Would be populated with actual changes
                    confidence=fix_data.get("confidence", 0.5),
                    risk_level="low"
                )
                candidates.append(candidate)

        return candidates[:self.max_fix_candidates]

    def _generate_attribute_error_fixes(self, error_analysis: Dict[str, Any]) -> List[FixCandidate]:
        """Generate fixes for AttributeError"""
        fixes = []

        variables = error_analysis.get("variables_involved", [])
        for var in variables:
            if var and var[0].islower():  # Likely a variable name
                fixes.append(FixCandidate(
                    description=f"Add hasattr check for {var}",
                    code_changes=[],
                    confidence=0.8,
                    risk_level="low",
                    test_cases=[f"Test with {var} = None", f"Test with missing attribute"]
                ))

                fixes.append(FixCandidate(
                    description=f"Use getattr with default for {var}",
                    code_changes=[],
                    confidence=0.7,
                    risk_level="low",
                    test_cases=[f"Test getattr fallback for {var}"]
                ))

        return fixes

    def _generate_key_error_fixes(self, error_analysis: Dict[str, Any]) -> List[FixCandidate]:
        """Generate fixes for KeyError"""
        fixes = []

        variables = error_analysis.get("variables_involved", [])
        for var in variables:
            fixes.append(FixCandidate(
                description=f"Use dict.get() for safe access to {var}",
                code_changes=[],
                confidence=0.9,
                risk_level="low",
                test_cases=[f"Test with missing key in {var}"]
            ))

            fixes.append(FixCandidate(
                description=f"Check if key exists in {var} before access",
                code_changes=[],
                confidence=0.8,
                risk_level="low",
                test_cases=[f"Test key existence check for {var}"]
            ))

        return fixes

    def _generate_type_error_fixes(self, error_analysis: Dict[str, Any]) -> List[FixCandidate]:
        """Generate fixes for TypeError"""
        fixes = []

        error_message = error_analysis.get("error_message", "").lower()

        if "none" in error_message and ("+" in error_message or "-" in error_message):
            fixes.append(FixCandidate(
                description="Add None check before arithmetic operation",
                code_changes=[],
                confidence=0.8,
                risk_level="low",
                test_cases=["Test with None values in operations"]
            ))

        if "int()" in error_message or "float()" in error_message or "str()" in error_message:
            fixes.append(FixCandidate(
                description="Add try-except for type conversion",
                code_changes=[],
                confidence=0.7,
                risk_level="low",
                test_cases=["Test type conversion with invalid inputs"]
            ))

        return fixes

    def _generate_import_error_fixes(self, error_analysis: Dict[str, Any]) -> List[FixCandidate]:
        """Generate fixes for ImportError"""
        fixes = []

        error_message = error_analysis.get("error_message", "")
        missing_module = ""

        # Try to extract module name
        if "No module named" in error_message:
            import re
            match = re.search(r"No module named ['\"]([^'\"]+)['\"]", error_message)
            if match:
                missing_module = match.group(1)

        if missing_module:
            fixes.append(FixCandidate(
                description=f"Install missing module: pip install {missing_module}",
                code_changes=[],
                confidence=0.9,
                risk_level="low",
                test_cases=[f"Test import after installing {missing_module}"]
            ))

            fixes.append(FixCandidate(
                description=f"Add try-except import for {missing_module}",
                code_changes=[],
                confidence=0.8,
                risk_level="low",
                test_cases=[f"Test graceful handling when {missing_module} unavailable"]
            ))

        return fixes

    def _generate_value_error_fixes(self, error_analysis: Dict[str, Any]) -> List[FixCandidate]:
        """Generate fixes for ValueError"""
        fixes = []

        error_message = error_analysis.get("error_message", "").lower()

        if "invalid" in error_message and "literal" in error_message:
            fixes.append(FixCandidate(
                description="Add input validation before conversion",
                code_changes=[],
                confidence=0.7,
                risk_level="low",
                test_cases=["Test with invalid input values"]
            ))

        return fixes

    def _generate_io_error_fixes(self, error_analysis: Dict[str, Any]) -> List[FixCandidate]:
        """Generate fixes for IOError/FileNotFoundError"""
        fixes = []

        fixes.append(FixCandidate(
            description="Add file existence check before access",
            code_changes=[],
            confidence=0.8,
            risk_level="low",
            test_cases=["Test with non-existent files", "Test with permission issues"]
        ))

        fixes.append(FixCandidate(
            description="Add try-except for file operations",
            code_changes=[],
            confidence=0.7,
            risk_level="low",
            test_cases=["Test file operation error handling"]
        ))

        return fixes

    def _generate_generic_fixes(self, error_analysis: Dict[str, Any]) -> List[FixCandidate]:
        """Generate generic error handling fixes"""
        return [
            FixCandidate(
                description="Add comprehensive try-except block",
                code_changes=[],
                confidence=0.6,
                risk_level="medium",
                test_cases=["Test error handling paths"]
            ),
            FixCandidate(
                description="Add logging for error diagnosis",
                code_changes=[],
                confidence=0.7,
                risk_level="low",
                test_cases=["Verify error logging works"]
            ),
            FixCandidate(
                description="Add input validation",
                code_changes=[],
                confidence=0.5,
                risk_level="low",
                test_cases=["Test with invalid inputs"]
            )
        ]

    async def _validate_fix_candidates(self, candidates: List[FixCandidate],
                                      error_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate and rank fix candidates"""
        validated = []

        for candidate in candidates:
            # Basic validation
            validation = self.fix_validator.validate_fix(
                error_analysis.get("code_context", ""),
                {"code_changes": candidate.code_changes}
            )

            validated_candidate = {
                "description": candidate.description,
                "confidence": candidate.confidence,
                "risk_level": candidate.risk_level,
                "validation": validation,
                "test_cases": candidate.test_cases,
                "rollback_plan": candidate.rollback_plan,
                "overall_score": self._calculate_fix_score(candidate, validation)
            }

            validated.append(validated_candidate)

        # Sort by overall score
        validated.sort(key=lambda x: x["overall_score"], reverse=True)

        return validated

    def _calculate_fix_score(self, candidate: FixCandidate, validation: Dict[str, Any]) -> float:
        """Calculate overall score for a fix candidate"""
        score = candidate.confidence * 0.6  # Base confidence

        # Validation bonus
        if validation.get("is_valid", False):
            score += 0.2

        # Risk penalty
        risk_level = candidate.risk_level
        if risk_level == "low":
            score += 0.1
        elif risk_level == "high":
            score -= 0.2

        # Test coverage bonus
        if candidate.test_cases:
            score += min(0.1, len(candidate.test_cases) * 0.02)

        return max(0, min(1, score))

    def _select_best_fix(self, validated_candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select the best fix from validated candidates"""
        if not validated_candidates:
            return None

        # Filter by minimum confidence
        suitable_candidates = [
            candidate for candidate in validated_candidates
            if candidate["overall_score"] >= self.min_confidence_threshold
        ]

        if not suitable_candidates:
            return None

        return suitable_candidates[0]

    async def _generate_fix_implementation(self, fix: Dict[str, Any],
                                         error_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed fix implementation"""
        implementation = {
            "type": "code_fix",
            "description": fix["description"],
            "code_changes": [],
            "test_commands": [],
            "validation_steps": []
        }

        # Generate specific implementation based on fix type
        error_category = error_analysis.get("error_category")

        if error_category == "attribute_error":
            implementation["code_changes"] = self._implement_attribute_error_fix(error_analysis)
        elif error_category == "key_error":
            implementation["code_changes"] = self._implement_key_error_fix(error_analysis)
        elif error_category == "type_error":
            implementation["code_changes"] = self._implement_type_error_fix(error_analysis)

        # Generate test commands
        implementation["test_commands"] = [
            f"python -c \"{'import sys; ' if 'import' in fix['description'] else ''}exec(open('{error_analysis.get('file_path', 'test.py')}').read())\""
        ]

        # Generate validation steps
        implementation["validation_steps"] = [
            "Run the fixed code",
            "Check for new errors",
            "Run existing tests",
            "Verify fix addresses original error"
        ]

        return implementation

    def _implement_attribute_error_fix(self, error_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Implement attribute error fix"""
        return [{
            "file_path": error_analysis.get("file_path", ""),
            "line_number": error_analysis.get("line_number", 1),
            "change_type": "add_check",
            "code": "if hasattr(obj, 'attribute'): ..."
        }]

    def _implement_key_error_fix(self, error_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Implement key error fix"""
        return [{
            "file_path": error_analysis.get("file_path", ""),
            "line_number": error_analysis.get("line_number", 1),
            "change_type": "safe_access",
            "code": "value = dict.get('key', default)"
        }]

    def _implement_type_error_fix(self, error_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Implement type error fix"""
        return [{
            "file_path": error_analysis.get("file_path", ""),
            "line_number": error_analysis.get("line_number", 1),
            "change_type": "type_check",
            "code": "if value is not None: ..."
        }]

    async def apply_fix(self, fix_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a generated fix"""
        try:
            implementation = fix_result.get("fix", {})
            code_changes = implementation.get("code_changes", [])

            if not code_changes:
                return {"success": False, "error": "No code changes to apply"}

            # Apply each change
            applied_changes = []
            for change in code_changes:
                file_path = change.get("file_path")
                if file_path and os.path.exists(file_path):
                    result = self.fix_applier.apply_fix(file_path, change)
                    applied_changes.append(result)

            # Update fix history
            fix_record = next((f for f in self.fix_history if f["timestamp"] == fix_result.get("timestamp")), None)
            if fix_record:
                fix_record["success"] = all(change.get("success", False) for change in applied_changes)

            return {
                "success": True,
                "changes_applied": len(applied_changes),
                "backup_created": any(change.get("backup_path") for change in applied_changes),
                "rollback_available": True
            }

        except Exception as e:
            self.logger.error(f"Error applying fix: {e}")
            return {"success": False, "error": str(e)}

    async def _load_fix_database(self):
        """Load fix patterns and history"""
        try:
            # Load successful fixes
            success_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'successful_fixes.json')
            if os.path.exists(success_file):
                with open(success_file, 'r') as f:
                    self.successful_fixes = json.load(f)

            # Load failed fixes
            failed_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'failed_fixes.json')
            if os.path.exists(failed_file):
                with open(failed_file, 'r') as f:
                    self.failed_fixes = json.load(f)

        except Exception as e:
            self.logger.debug(f"Could not load fix database: {e}")

    def get_fix_stats(self) -> Dict[str, Any]:
        """Get fix generation statistics"""
        total_fixes = len(self.fix_history)
        successful_fixes = sum(1 for f in self.fix_history if f.get("success") is True)

        return {
            "total_fixes_generated": total_fixes,
            "successful_fixes": successful_fixes,
            "success_rate": successful_fixes / total_fixes if total_fixes > 0 else 0,
            "average_confidence": sum(f.get("confidence", 0) for f in self.fix_history) / total_fixes if total_fixes > 0 else 0
        }

    async def shutdown(self):
        """Shutdown the advanced fix generator"""
        try:
            self.logger.info("Shutting down advanced fix generator...")

            # Save fix database
            await self._save_fix_database()

            self.logger.info("Advanced fix generator shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down fix generator: {e}")

    async def _save_fix_database(self):
        """Save fix patterns and history"""
        try:
            # Save successful fixes
            success_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'successful_fixes.json')
            os.makedirs(os.path.dirname(success_file), exist_ok=True)

            with open(success_file, 'w') as f:
                json.dump(self.successful_fixes, f, indent=2)

            # Save failed fixes
            failed_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'failed_fixes.json')

            with open(failed_file, 'w') as f:
                json.dump(self.failed_fixes, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving fix database: {e}")