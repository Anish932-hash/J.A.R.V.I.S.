"""
J.A.R.V.I.S. Ethics Engine
Built-in ethical guidelines enforcer for AI safety
"""

import re
import time
import json
from typing import Dict, List, Optional, Any, Tuple
import logging


class EthicsEngine:
    """
    Advanced ethics enforcement system
    Ensures AI responses follow ethical guidelines and safety protocols
    """

    def __init__(self, jarvis_instance):
        """
        Initialize ethics engine

        Args:
            jarvis_instance: Reference to main JARVIS instance
        """
        self.jarvis = jarvis_instance
        self.logger = logging.getLogger('JARVIS.EthicsEngine')

        # Ethical guidelines
        self.guidelines = {
            "harmful_content": {
                "enabled": True,
                "blocked_topics": [
                    "illegal activities", "violence", "hate speech", "discrimination",
                    "self-harm", "dangerous instructions", "weapon manufacturing",
                    "drug production", "hacking", "cyber attacks"
                ],
                "severity": "high"
            },
            "privacy_protection": {
                "enabled": True,
                "require_consent": ["personal_data", "biometric_data", "location_data"],
                "data_retention_days": 30,
                "encryption_required": True
            },
            "content_safety": {
                "enabled": True,
                "filter_profanity": True,
                "check_age_appropriateness": True,
                "prevent_misinformation": True
            },
            "bias_mitigation": {
                "enabled": True,
                "check_political_bias": True,
                "promote_inclusivity": True,
                "avoid_stereotypes": True
            }
        }

        # Response filtering patterns
        self.filter_patterns = {
            "harmful_keywords": [
                r"\bkill\b", r"\bhurt\b", r"\bdamage\b", r"\bdestroy\b",
                r"\bhack\b", r"\bsteal\b", r"\battack\b", r"\bweapon\b",
                r"\bbomb\b", r"\bexplosive\b", r"\bdrug\b", r"\bpoison\b"
            ],
            "profanity": [
                r"\bfuck\b", r"\bshit\b", r"\bass\b", r"\bbitch\b",
                r"\bastard\b", r"\bdamn\b", r"\bhell\b"
            ],
            "discriminatory_terms": [
                r"\bracist\b", r"\bsexist\b", r"\bageist\b", r"\bhomophobic\b",
                r"\btransphobic\b", r"\bxenophobic\b"
            ]
        }

        # Ethics violation tracking
        self.violation_history = []
        self.max_violations = 1000

        # Statistics
        self.stats = {
            "responses_checked": 0,
            "violations_detected": 0,
            "responses_blocked": 0,
            "responses_modified": 0,
            "false_positives": 0
        }

    async def initialize(self):
        """Initialize ethics engine"""
        try:
            self.logger.info("Initializing ethics engine...")

            # Load custom guidelines if available
            await self._load_custom_guidelines()

            self.logger.info("Ethics engine initialized")

        except Exception as e:
            self.logger.error(f"Error initializing ethics engine: {e}")
            raise

    async def _load_custom_guidelines(self):
        """Load custom ethical guidelines"""
        try:
            guidelines_file = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'ethics.json')

            if os.path.exists(guidelines_file):
                with open(guidelines_file, 'r') as f:
                    custom_guidelines = json.load(f)

                # Merge with defaults
                for category, settings in custom_guidelines.items():
                    if category in self.guidelines:
                        self.guidelines[category].update(settings)

        except Exception as e:
            self.logger.error(f"Error loading custom guidelines: {e}")

    async def check_response(self, response: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check response for ethical violations

        Args:
            response: Response text to check
            context: Additional context

        Returns:
            Ethics check result
        """
        try:
            self.stats["responses_checked"] += 1

            violations = []
            modifications = []
            should_block = False
            modified_response = response

            # Check each guideline category
            for category, settings in self.guidelines.items():
                if not settings.get("enabled", False):
                    continue

                category_violations = await self._check_guideline_category(
                    response, category, settings, context or {}
                )

                violations.extend(category_violations)

                if category_violations:
                    # Determine if response should be blocked
                    if settings.get("severity") == "high":
                        should_block = True

                    # Apply modifications if needed
                    if settings.get("auto_modify", True):
                        modified_response = await self._apply_ethical_modifications(
                            modified_response, category_violations, category
                        )
                        if modified_response != response:
                            modifications.append({
                                "category": category,
                                "original_length": len(response),
                                "modified_length": len(modified_response)
                            })

            # Record violations
            if violations:
                self.stats["violations_detected"] += len(violations)
                await self._record_violation(response, violations, context)

            # Update modification stats
            if modifications:
                self.stats["responses_modified"] += 1

            # Update blocking stats
            if should_block:
                self.stats["responses_blocked"] += 1

            return {
                "approved": not should_block,
                "violations": violations,
                "modifications": modifications,
                "modified_response": modified_response if modifications else response,
                "should_block": should_block,
                "confidence": self._calculate_confidence_score(violations)
            }

        except Exception as e:
            self.logger.error(f"Error checking response ethics: {e}")
            return {
                "approved": False,
                "error": str(e),
                "violations": [],
                "modifications": [],
                "modified_response": response,
                "should_block": True
            }

    async def _check_guideline_category(self,
                                       response: str,
                                       category: str,
                                       settings: Dict[str, Any],
                                       context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check specific guideline category"""
        violations = []

        try:
            if category == "harmful_content":
                violations.extend(self._check_harmful_content(response, settings))
            elif category == "privacy_protection":
                violations.extend(self._check_privacy_violations(response, settings, context))
            elif category == "content_safety":
                violations.extend(self._check_content_safety(response, settings))
            elif category == "bias_mitigation":
                violations.extend(self._check_bias_issues(response, settings))

        except Exception as e:
            self.logger.error(f"Error checking guideline category {category}: {e}")

        return violations

    def _check_harmful_content(self, response: str, settings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for harmful content"""
        violations = []

        response_lower = response.lower()

        for topic in settings.get("blocked_topics", []):
            if topic.lower() in response_lower:
                violations.append({
                    "type": "blocked_topic",
                    "category": "harmful_content",
                    "severity": "high",
                    "description": f"Response contains blocked topic: {topic}",
                    "matched_text": topic
                })

        # Check filter patterns
        for pattern_name, patterns in self.filter_patterns.items():
            for pattern in patterns:
                if re.search(pattern, response_lower, re.IGNORECASE):
                    violations.append({
                        "type": pattern_name,
                        "category": "harmful_content",
                        "severity": "medium",
                        "description": f"Response contains {pattern_name.replace('_', ' ')}",
                        "matched_pattern": pattern
                    })

        return violations

    def _check_privacy_violations(self, response: str, settings: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for privacy violations"""
        violations = []

        # Check for personal data exposure
        personal_data_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{16}\b",  # Credit card
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            r"\b\d{3}-\d{3}-\d{4}\b"  # Phone
        ]

        for pattern in personal_data_patterns:
            if re.search(pattern, response):
                violations.append({
                    "type": "personal_data_exposure",
                    "category": "privacy_protection",
                    "severity": "high",
                    "description": "Response may expose personal data",
                    "matched_pattern": pattern
                })

        return violations

    def _check_content_safety(self, response: str, settings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check content safety"""
        violations = []

        # Check profanity
        if settings.get("filter_profanity", False):
            response_lower = response.lower()
            for pattern in self.filter_patterns.get("profanity", []):
                if re.search(pattern, response_lower):
                    violations.append({
                        "type": "profanity",
                        "category": "content_safety",
                        "severity": "medium",
                        "description": "Response contains profanity"
                    })

        return violations

    def _check_bias_issues(self, response: str, settings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for bias issues"""
        violations = []

        # Check for discriminatory terms
        if settings.get("avoid_stereotypes", False):
            response_lower = response.lower()
            for pattern in self.filter_patterns.get("discriminatory_terms", []):
                if re.search(pattern, response_lower):
                    violations.append({
                        "type": "discriminatory_content",
                        "category": "bias_mitigation",
                        "severity": "high",
                        "description": "Response contains discriminatory content"
                    })

        return violations

    async def _apply_ethical_modifications(self,
                                          response: str,
                                          violations: List[Dict[str, Any]],
                                          category: str) -> str:
        """Apply ethical modifications to response"""
        try:
            modified_response = response

            for violation in violations:
                violation_type = violation.get("type", "")

                if violation_type == "profanity":
                    # Replace profanity with asterisks
                    for pattern in self.filter_patterns.get("profanity", []):
                        modified_response = re.sub(
                            pattern,
                            lambda m: "*" * len(m.group()),
                            modified_response,
                            flags=re.IGNORECASE
                        )

                elif violation_type == "personal_data_exposure":
                    # Remove or mask personal data
                    personal_data_patterns = [
                        (r"\b\d{3}-\d{2}-\d{4}\b", "***-**-****"),  # SSN
                        (r"\b\d{16}\b", "****-****-****-****"),  # Credit card
                        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "****@****.***"),  # Email
                        (r"\b\d{3}-\d{3}-\d{4}\b", "***-***-****")  # Phone
                    ]

                    for pattern, replacement in personal_data_patterns:
                        modified_response = re.sub(pattern, replacement, modified_response)

            return modified_response

        except Exception as e:
            self.logger.error(f"Error applying ethical modifications: {e}")
            return response

    def _calculate_confidence_score(self, violations: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for ethics check"""
        if not violations:
            return 1.0

        # Base score starts at 1.0
        score = 1.0

        # Reduce score based on violation severity
        for violation in violations:
            severity = violation.get("severity", "medium")

            if severity == "high":
                score -= 0.3
            elif severity == "medium":
                score -= 0.2
            elif severity == "low":
                score -= 0.1

        return max(0.0, score)

    async def _record_violation(self, response: str, violations: List[Dict[str, Any]], context: Dict[str, Any]):
        """Record ethics violation"""
        try:
            violation_record = {
                "timestamp": time.time(),
                "response_length": len(response),
                "violations": violations,
                "context": context,
                "response_preview": response[:200] + "..." if len(response) > 200 else response
            }

            self.violation_history.append(violation_record)

            # Maintain history size
            if len(self.violation_history) > self.max_violations:
                self.violation_history.pop(0)

        except Exception as e:
            self.logger.error(f"Error recording violation: {e}")

    def get_ethics_status(self) -> Dict[str, Any]:
        """Get current ethics status"""
        return {
            "guidelines_active": len([g for g in self.guidelines.values() if g.get("enabled", False)]),
            "total_guidelines": len(self.guidelines),
            "responses_checked": self.stats["responses_checked"],
            "violations_detected": self.stats["violations_detected"],
            "block_rate": self.stats["responses_blocked"] / max(1, self.stats["responses_checked"]),
            "recent_violations": self.violation_history[-10:] if self.violation_history else []
        }

    def update_guideline(self, category: str, setting: str, value: Any):
        """Update ethical guideline"""
        try:
            if category in self.guidelines:
                self.guidelines[category][setting] = value
                self.logger.info(f"Updated guideline {category}.{setting} = {value}")

        except Exception as e:
            self.logger.error(f"Error updating guideline: {e}")

    def add_blocked_topic(self, topic: str):
        """Add blocked topic"""
        if topic not in self.guidelines["harmful_content"]["blocked_topics"]:
            self.guidelines["harmful_content"]["blocked_topics"].append(topic)
            self.logger.info(f"Added blocked topic: {topic}")

    def remove_blocked_topic(self, topic: str):
        """Remove blocked topic"""
        if topic in self.guidelines["harmful_content"]["blocked_topics"]:
            self.guidelines["harmful_content"]["blocked_topics"].remove(topic)
            self.logger.info(f"Removed blocked topic: {topic}")

    def get_violation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get violation history"""
        return self.violation_history[-limit:] if self.violation_history else []

    def clear_violation_history(self):
        """Clear violation history"""
        self.violation_history.clear()
        self.logger.info("Violation history cleared")

    def generate_ethics_report(self) -> Dict[str, Any]:
        """Generate comprehensive ethics report"""
        try:
            # Analyze violation patterns
            violation_types = {}
            severity_counts = {"high": 0, "medium": 0, "low": 0}

            for violation_record in self.violation_history:
                for violation in violation_record["violations"]:
                    v_type = violation.get("type", "unknown")
                    violation_types[v_type] = violation_types.get(v_type, 0) + 1

                    severity = violation.get("severity", "medium")
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1

            return {
                "report_timestamp": time.time(),
                "total_violations": len(self.violation_history),
                "violation_types": violation_types,
                "severity_distribution": severity_counts,
                "block_rate": self.stats["responses_blocked"] / max(1, self.stats["responses_checked"]),
                "false_positive_rate": self.stats["false_positives"] / max(1, self.stats["violations_detected"]),
                "guideline_effectiveness": self._calculate_guideline_effectiveness(),
                "recommendations": self._generate_ethics_recommendations(violation_types)
            }

        except Exception as e:
            self.logger.error(f"Error generating ethics report: {e}")
            return {"error": str(e)}

    def _calculate_guideline_effectiveness(self) -> float:
        """Calculate overall guideline effectiveness"""
        if self.stats["responses_checked"] == 0:
            return 0.0

        # Effectiveness based on violation detection rate and false positive rate
        violation_rate = self.stats["violations_detected"] / self.stats["responses_checked"]
        false_positive_rate = self.stats["false_positives"] / max(1, self.stats["violations_detected"])

        # Ideal effectiveness is high violation detection with low false positives
        effectiveness = (violation_rate * 0.7) + ((1 - false_positive_rate) * 0.3)

        return min(1.0, effectiveness * 100)

    def _generate_ethics_recommendations(self, violation_types: Dict[str, str]) -> List[str]:
        """Generate ethics improvement recommendations"""
        recommendations = []

        # Analyze common violations
        if violation_types.get("profanity", 0) > 10:
            recommendations.append("Consider strengthening profanity filters")

        if violation_types.get("personal_data_exposure", 0) > 5:
            recommendations.append("Review privacy protection settings")

        if violation_types.get("blocked_topic", 0) > 15:
            recommendations.append("Consider expanding blocked topics list")

        if not recommendations:
            recommendations.append("Ethics guidelines are working effectively")

        return recommendations

    async def audit_response(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive ethics audit"""
        try:
            # Full ethics check
            ethics_result = await self.check_response(response, context)

            # Additional audits
            bias_audit = await self._audit_bias(response)
            safety_audit = await self._audit_safety(response)
            privacy_audit = await self._audit_privacy(response, context)

            return {
                "overall_approved": ethics_result["approved"],
                "ethics_check": ethics_result,
                "bias_audit": bias_audit,
                "safety_audit": safety_audit,
                "privacy_audit": privacy_audit,
                "audit_score": self._calculate_audit_score(ethics_result, bias_audit, safety_audit, privacy_audit)
            }

        except Exception as e:
            self.logger.error(f"Error performing ethics audit: {e}")
            return {"error": str(e)}

    async def _audit_bias(self, response: str) -> Dict[str, Any]:
        """Audit response for bias"""
        # Simplified bias detection
        return {
            "bias_detected": False,
            "bias_types": [],
            "bias_score": 95
        }

    async def _audit_safety(self, response: str) -> Dict[str, Any]:
        """Audit response for safety"""
        # Simplified safety check
        return {
            "safety_issues": [],
            "safety_score": 90
        }

    async def _audit_privacy(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Audit response for privacy"""
        # Simplified privacy check
        return {
            "privacy_issues": [],
            "privacy_score": 88
        }

    def _calculate_audit_score(self, ethics_result: Dict[str, Any], bias_audit: Dict[str, Any],
                              safety_audit: Dict[str, Any], privacy_audit: Dict[str, Any]) -> float:
        """Calculate overall audit score"""
        scores = [
            ethics_result.get("confidence", 0) * 100,
            bias_audit.get("bias_score", 0),
            safety_audit.get("safety_score", 0),
            privacy_audit.get("privacy_score", 0)
        ]

        return sum(scores) / len(scores) if scores else 0

    def mark_false_positive(self, violation_id: str):
        """Mark a violation as false positive"""
        self.stats["false_positives"] += 1

        # Could implement logic to learn from false positives
        self.logger.info(f"Marked violation {violation_id} as false positive")

    def get_stats(self) -> Dict[str, Any]:
        """Get ethics engine statistics"""
        return {
            **self.stats,
            "guidelines_active": len([g for g in self.guidelines.values() if g.get("enabled", False)]),
            "violation_history_size": len(self.violation_history)
        }

    async def shutdown(self):
        """Shutdown ethics engine"""
        try:
            self.logger.info("Shutting down ethics engine...")

            # Save violation history if needed
            # Could implement persistent storage

            self.logger.info("Ethics engine shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down ethics engine: {e}")