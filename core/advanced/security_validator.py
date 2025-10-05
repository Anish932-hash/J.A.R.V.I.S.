"""
J.A.R.V.I.S. Security Validator
Advanced security validation and vulnerability assessment for AI-generated code
"""

import os
import re
import ast
import logging
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import asyncio

# Security analysis imports
try:
    import bandit
    BANDIT_AVAILABLE = True
except ImportError:
    BANDIT_AVAILABLE = False


class SecurityValidator:
    """
    Ultra-advanced security validator that analyzes code for vulnerabilities,
    validates security practices, and ensures AI-generated code is secure
    """

    def __init__(self, development_engine):
        """
        Initialize security validator

        Args:
            development_engine: Reference to self-development engine
        """
        self.engine = development_engine
        self.jarvis = development_engine.jarvis
        self.logger = logging.getLogger('JARVIS.SecurityValidator')

        # Security rules and patterns
        self.security_rules = self._load_security_rules()
        self.vulnerability_patterns = self._load_vulnerability_patterns()

        # Security statistics
        self.stats = {
            'validations_performed': 0,
            'vulnerabilities_found': 0,
            'security_score_avg': 0.0,
            'high_risk_issues': 0,
            'critical_issues': 0,
            'validation_time': 0
        }

    async def initialize(self):
        """Initialize security validator"""
        try:
            self.logger.info("Initializing security validator...")

            # Validate security rules integrity
            await self._validate_security_rules()

            self.logger.info("Security validator initialized")

        except Exception as e:
            self.logger.error(f"Error initializing security validator: {e}")
            raise

    async def validate_code_security(self,
                                   code: str,
                                   context: Dict[str, Any] = None,
                                   security_level: str = "standard") -> Dict[str, Any]:
        """
        Comprehensive security validation of code

        Args:
            code: Code to validate
            context: Contextual information
            security_level: Security validation level (basic, standard, strict)

        Returns:
            Security validation results
        """
        start_time = asyncio.get_event_loop().time()

        try:
            self.logger.info(f"Performing {security_level} security validation")

            # Parse code for analysis
            code_analysis = await self._analyze_code_ast(code)

            # Apply security rules
            rule_violations = await self._apply_security_rules(code, code_analysis, security_level)

            # Detect vulnerabilities
            vulnerabilities = await self._detect_vulnerabilities(code, code_analysis)

            # Assess security posture
            security_assessment = await self._assess_security_posture(code, rule_violations, vulnerabilities)

            # Generate recommendations
            recommendations = await self._generate_security_recommendations(rule_violations, vulnerabilities)

            # Calculate security score
            security_score = self._calculate_security_score(rule_violations, vulnerabilities, security_level)

            validation_time = asyncio.get_event_loop().time() - start_time
            self.stats['validation_time'] += validation_time
            self.stats['validations_performed'] += 1

            result = {
                'code_hash': hashlib.md5(code.encode()).hexdigest(),
                'security_level': security_level,
                'security_score': security_score,
                'is_secure': security_score >= self._get_security_threshold(security_level),
                'rule_violations': rule_violations,
                'vulnerabilities': vulnerabilities,
                'security_assessment': security_assessment,
                'recommendations': recommendations,
                'validation_time': validation_time,
                'timestamp': asyncio.get_event_loop().time()
            }

            # Update statistics
            if vulnerabilities:
                self.stats['vulnerabilities_found'] += len(vulnerabilities)
                high_risk = sum(1 for v in vulnerabilities if v.get('severity') in ['high', 'critical'])
                self.stats['high_risk_issues'] += high_risk
                self.stats['critical_issues'] += sum(1 for v in vulnerabilities if v.get('severity') == 'critical')

            self.logger.info(f"Security validation completed with score: {security_score:.2f}")
            return result

        except Exception as e:
            self.logger.error(f"Error in security validation: {e}")
            return {
                'error': str(e),
                'is_secure': False,
                'security_score': 0.0,
                'validation_time': asyncio.get_event_loop().time() - start_time
            }

    async def _analyze_code_ast(self, code: str) -> Dict[str, Any]:
        """Analyze code AST for security-relevant structures"""
        analysis = {
            'imports': [],
            'function_calls': [],
            'variable_assignments': [],
            'string_literals': [],
            'exec_patterns': [],
            'network_operations': [],
            'file_operations': [],
            'data_flow': []
        }

        try:
            tree = ast.parse(code)

            # Extract security-relevant information
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    analysis['imports'].extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    analysis['imports'].extend(f"{module}.{alias.name}" for alias in node.names)

                elif isinstance(node, ast.Call):
                    func_name = self._get_function_name(node)
                    if func_name:
                        analysis['function_calls'].append({
                            'function': func_name,
                            'line': getattr(node, 'lineno', 0),
                            'args': len(node.args)
                        })

                elif isinstance(node, ast.Str):
                    analysis['string_literals'].append({
                        'value': node.s,
                        'line': getattr(node, 'lineno', 0)
                    })

                elif isinstance(node, ast.Exec):
                    analysis['exec_patterns'].append({
                        'type': 'exec_statement',
                        'line': getattr(node, 'lineno', 0)
                    })

            # Analyze data flow
            analysis['data_flow'] = self._analyze_data_flow(tree)

        except SyntaxError:
            analysis['syntax_errors'] = True

        return analysis

    def _get_function_name(self, call_node: ast.Call) -> str:
        """Get function name from call node"""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            return f"{self._get_attribute_name(call_node.func)}"
        return ""

    def _get_attribute_name(self, attr_node: ast.Attribute) -> str:
        """Get full attribute name"""
        if isinstance(attr_node.value, ast.Name):
            return f"{attr_node.value.id}.{attr_node.attr}"
        elif isinstance(attr_node.value, ast.Attribute):
            return f"{self._get_attribute_name(attr_node.value)}.{attr_node.attr}"
        return attr_node.attr

    def _analyze_data_flow(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze data flow for security issues"""
        data_flow = []

        # Simple data flow analysis
        variables = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        if isinstance(node.value, ast.Str):
                            variables[var_name] = 'string'
                        elif isinstance(node.value, ast.Call):
                            func_name = self._get_function_name(node.value)
                            variables[var_name] = f'call_result_{func_name}'

            elif isinstance(node, ast.Call):
                func_name = self._get_function_name(node)
                if func_name in ['eval', 'exec', 'subprocess.call']:
                    for arg in node.args:
                        if isinstance(arg, ast.Name) and arg.id in variables:
                            data_flow.append({
                                'type': 'dangerous_call',
                                'function': func_name,
                                'variable': arg.id,
                                'variable_type': variables[arg.id],
                                'line': getattr(node, 'lineno', 0)
                            })

        return data_flow

    async def _apply_security_rules(self, code: str, code_analysis: Dict[str, Any], security_level: str) -> List[Dict[str, Any]]:
        """Apply security rules to code"""
        violations = []

        # Apply each security rule
        for rule_name, rule_config in self.security_rules.items():
            if self._rule_applies(rule_config, security_level):
                rule_violations = await self._check_rule(code, code_analysis, rule_name, rule_config)
                violations.extend(rule_violations)

        return violations

    def _rule_applies(self, rule_config: Dict[str, Any], security_level: str) -> bool:
        """Check if rule applies to current security level"""
        rule_levels = rule_config.get('levels', ['basic', 'standard', 'strict'])
        return security_level in rule_levels

    async def _check_rule(self, code: str, code_analysis: Dict[str, Any], rule_name: str, rule_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check a specific security rule"""
        violations = []

        pattern = rule_config.get('pattern', '')
        if pattern:
            matches = re.findall(pattern, code, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                violations.append({
                    'rule': rule_name,
                    'type': rule_config.get('type', 'warning'),
                    'severity': rule_config.get('severity', 'medium'),
                    'description': rule_config.get('description', ''),
                    'match': match,
                    'recommendation': rule_config.get('recommendation', '')
                })

        # Check AST-based rules
        ast_checks = rule_config.get('ast_checks', [])
        for check in ast_checks:
            check_type = check.get('type')
            if check_type == 'dangerous_import':
                dangerous_imports = check.get('imports', [])
                for imp in code_analysis.get('imports', []):
                    if any(dangerous in imp for dangerous in dangerous_imports):
                        violations.append({
                            'rule': rule_name,
                            'type': 'security_violation',
                            'severity': check.get('severity', 'high'),
                            'description': f"Dangerous import detected: {imp}",
                            'recommendation': check.get('recommendation', 'Avoid using this module')
                        })

        return violations

    async def _detect_vulnerabilities(self, code: str, code_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect specific vulnerabilities in code"""
        vulnerabilities = []

        # Check vulnerability patterns
        for vuln_name, vuln_config in self.vulnerability_patterns.items():
            pattern = vuln_config.get('pattern', '')
            if pattern:
                matches = re.findall(pattern, code, re.MULTILINE | re.DOTALL)
                for match in matches:
                    vulnerabilities.append({
                        'vulnerability': vuln_name,
                        'type': vuln_config.get('type', 'code_injection'),
                        'severity': vuln_config.get('severity', 'high'),
                        'description': vuln_config.get('description', ''),
                        'cwe': vuln_config.get('cwe', ''),
                        'match': match,
                        'recommendation': vuln_config.get('recommendation', '')
                    })

        # Check for data flow vulnerabilities
        for flow in code_analysis.get('data_flow', []):
            if flow['type'] == 'dangerous_call':
                vulnerabilities.append({
                    'vulnerability': 'dynamic_code_execution',
                    'type': 'code_injection',
                    'severity': 'critical',
                    'description': f"Dynamic code execution via {flow['function']}",
                    'cwe': 'CWE-95',
                    'line': flow['line'],
                    'recommendation': 'Avoid dynamic code execution. Use safe alternatives.'
                })

        # Check for hardcoded secrets
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']'
        ]

        for pattern in secret_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            for match in matches:
                vulnerabilities.append({
                    'vulnerability': 'hardcoded_secret',
                    'type': 'information_disclosure',
                    'severity': 'high',
                    'description': 'Hardcoded secret detected',
                    'cwe': 'CWE-798',
                    'match': match,
                    'recommendation': 'Use environment variables or secure credential storage'
                })

        return vulnerabilities

    async def _assess_security_posture(self, code: str, rule_violations: List[Dict[str, Any]], vulnerabilities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall security posture"""
        assessment = {
            'overall_risk': 'low',
            'critical_issues': 0,
            'high_risk_issues': 0,
            'medium_risk_issues': 0,
            'low_risk_issues': 0,
            'risk_factors': [],
            'strengths': []
        }

        # Count issues by severity
        all_issues = rule_violations + vulnerabilities
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}

        for issue in all_issues:
            severity = issue.get('severity', 'medium')
            severity_counts[severity] += 1

        assessment.update(severity_counts)

        # Determine overall risk
        if severity_counts['critical'] > 0:
            assessment['overall_risk'] = 'critical'
        elif severity_counts['high'] > 2:
            assessment['overall_risk'] = 'high'
        elif severity_counts['high'] > 0 or severity_counts['medium'] > 3:
            assessment['overall_risk'] = 'medium'
        else:
            assessment['overall_risk'] = 'low'

        # Identify risk factors
        if severity_counts['critical'] > 0:
            assessment['risk_factors'].append('Critical security vulnerabilities present')
        if any('injection' in str(v) for v in vulnerabilities):
            assessment['risk_factors'].append('Code injection vulnerabilities detected')
        if any('hardcoded' in str(v) for v in vulnerabilities):
            assessment['risk_factors'].append('Hardcoded secrets found')

        # Identify strengths
        if severity_counts['critical'] == 0 and severity_counts['high'] == 0:
            assessment['strengths'].append('No critical or high-severity issues')
        if not any('injection' in str(v) for v in vulnerabilities):
            assessment['strengths'].append('No code injection vulnerabilities detected')

        return assessment

    async def _generate_security_recommendations(self, rule_violations: List[Dict[str, Any]], vulnerabilities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate security recommendations"""
        recommendations = []

        # Group issues by type
        issue_types = {}
        for issue in rule_violations + vulnerabilities:
            issue_type = issue.get('type', 'unknown')
            if issue_type not in issue_types:
                issue_types[issue_type] = []
            issue_types[issue_type].append(issue)

        # Generate recommendations based on issue types
        for issue_type, issues in issue_types.items():
            if issue_type == 'code_injection':
                recommendations.append({
                    'priority': 'critical',
                    'action': 'eliminate_dynamic_code_execution',
                    'description': 'Remove all eval(), exec(), and subprocess calls with user input',
                    'implementation': 'Use safe APIs and input validation'
                })
            elif issue_type == 'information_disclosure':
                recommendations.append({
                    'priority': 'high',
                    'action': 'remove_hardcoded_secrets',
                    'description': 'Replace hardcoded secrets with environment variables',
                    'implementation': 'Use os.environ.get() and secure credential management'
                })
            elif issue_type == 'dangerous_import':
                recommendations.append({
                    'priority': 'medium',
                    'action': 'review_imports',
                    'description': 'Review and replace dangerous module imports',
                    'implementation': 'Use secure alternatives or sandboxing'
                })

        # General recommendations
        if not recommendations:
            recommendations.append({
                'priority': 'low',
                'action': 'maintain_security_practices',
                'description': 'Continue following secure coding practices',
                'implementation': 'Regular security reviews and updates'
            })

        return recommendations

    def _calculate_security_score(self, rule_violations: List[Dict[str, Any]], vulnerabilities: List[Dict[str, Any]], security_level: str) -> float:
        """Calculate security score"""
        base_score = 100.0

        # Penalty for each issue based on severity
        severity_penalties = {
            'critical': 25,
            'high': 15,
            'medium': 8,
            'low': 3,
            'info': 1
        }

        all_issues = rule_violations + vulnerabilities

        for issue in all_issues:
            severity = issue.get('severity', 'medium')
            penalty = severity_penalties.get(severity, 5)
            base_score -= penalty

        # Adjust for security level
        if security_level == 'strict':
            base_score *= 0.9  # Stricter requirements
        elif security_level == 'basic':
            base_score *= 1.1  # More lenient

        return max(0.0, min(100.0, base_score))

    def _get_security_threshold(self, security_level: str) -> float:
        """Get security threshold for level"""
        thresholds = {
            'basic': 60.0,
            'standard': 75.0,
            'strict': 90.0
        }
        return thresholds.get(security_level, 75.0)

    def _load_security_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load security rules"""
        return {
            'dangerous_functions': {
                'pattern': r'\b(eval|exec|subprocess\.call|subprocess\.Popen)\s*\(',
                'type': 'security_violation',
                'severity': 'critical',
                'description': 'Dangerous function call detected',
                'recommendation': 'Use safe alternatives and input validation',
                'levels': ['basic', 'standard', 'strict']
            },
            'weak_crypto': {
                'pattern': r'\b(md5|sha1)\s*\(',
                'type': 'weak_crypto',
                'severity': 'high',
                'description': 'Weak cryptographic function used',
                'recommendation': 'Use SHA-256 or stronger algorithms',
                'levels': ['standard', 'strict']
            },
            'dangerous_imports': {
                'ast_checks': [{
                    'type': 'dangerous_import',
                    'imports': ['pickle', 'marshal', 'shelve'],
                    'severity': 'medium',
                    'recommendation': 'Use safer serialization methods like JSON'
                }],
                'levels': ['standard', 'strict']
            },
            'sql_injection': {
                'pattern': r'(\w+)\.execute\s*\(\s*["\'].*?%.*["\']',
                'type': 'sql_injection',
                'severity': 'high',
                'description': 'Potential SQL injection vulnerability',
                'recommendation': 'Use parameterized queries',
                'levels': ['basic', 'standard', 'strict']
            }
        }

    def _load_vulnerability_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load vulnerability patterns"""
        return {
            'command_injection': {
                'pattern': r'subprocess\.(call|Popen|run)\s*\(\s*.*\+.*\)',
                'type': 'command_injection',
                'severity': 'critical',
                'description': 'Command injection vulnerability',
                'cwe': 'CWE-78',
                'recommendation': 'Use shlex.quote() and avoid shell=True'
            },
            'path_traversal': {
                'pattern': r'open\s*\(\s*.*\+\s*.*\)',
                'type': 'path_traversal',
                'severity': 'high',
                'description': 'Potential path traversal vulnerability',
                'cwe': 'CWE-22',
                'recommendation': 'Use os.path.join() and validate paths'
            },
            'xss_vulnerable': {
                'pattern': r'print\s*\(\s*.*request.*\)',
                'type': 'xss',
                'severity': 'medium',
                'description': 'Potential XSS vulnerability',
                'cwe': 'CWE-79',
                'recommendation': 'Escape HTML output properly'
            }
        }

    async def _validate_security_rules(self):
        """Validate integrity of security rules"""
        # Ensure all required fields are present
        required_fields = ['type', 'severity', 'description']

        for rule_name, rule_config in self.security_rules.items():
            for field in required_fields:
                if field not in rule_config:
                    self.logger.warning(f"Security rule {rule_name} missing field: {field}")

    def get_security_stats(self) -> Dict[str, Any]:
        """Get security validation statistics"""
        return {
            **self.stats,
            'avg_security_score': self.stats['security_score_avg'] / max(1, self.stats['validations_performed']),
            'rules_count': len(self.security_rules),
            'vulnerability_patterns': len(self.vulnerability_patterns)
        }

    async def shutdown(self):
        """Shutdown security validator"""
        try:
            self.logger.info("Shutting down security validator...")

            # Clear rules and patterns
            self.security_rules.clear()
            self.vulnerability_patterns.clear()

            self.logger.info("Security validator shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down security validator: {e}")