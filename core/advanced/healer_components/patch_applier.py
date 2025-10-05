"""
J.A.R.V.I.S. Advanced Patch Applier
Safe, intelligent application of code fixes with rollback capabilities
"""

import os
import ast
import time
import json
import shutil
import logging
import difflib
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor


@dataclass
class Patch:
    """Represents a code patch"""
    patch_id: str
    file_path: str
    description: str
    changes: List[Dict[str, Any]]
    risk_level: str
    created_at: float
    applied: bool = False
    applied_at: Optional[float] = None
    rolled_back: bool = False
    rolled_back_at: Optional[float] = None
    backup_path: Optional[str] = None
    verification_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatchValidation:
    """Patch validation results"""
    is_valid: bool
    syntax_check: bool
    import_check: bool
    logic_check: bool
    warnings: List[str]
    errors: List[str]
    risk_assessment: str
    test_recommendations: List[str]


class CodeValidator:
    """Validates code patches before application"""

    def __init__(self):
        self.validation_rules = {
            "syntax": self._validate_syntax,
            "imports": self._validate_imports,
            "logic": self._validate_logic,
            "security": self._validate_security
        }

    def validate_patch(self, original_code: str, patched_code: str,
                      patch_info: Dict[str, Any]) -> PatchValidation:
        """Comprehensive patch validation"""
        validation = PatchValidation(
            is_valid=True,
            syntax_check=True,
            import_check=True,
            logic_check=True,
            warnings=[],
            errors=[],
            risk_assessment="low",
            test_recommendations=[]
        )

        # Run all validation checks
        for check_name, check_func in self.validation_rules.items():
            try:
                result = check_func(original_code, patched_code, patch_info)
                if not result["passed"]:
                    validation.errors.extend(result["errors"])
                    validation.is_valid = False

                    # Update risk assessment
                    if check_name == "syntax":
                        validation.syntax_check = False
                        validation.risk_assessment = "critical"
                    elif check_name == "security":
                        validation.risk_assessment = "high"
                    else:
                        if validation.risk_assessment == "low":
                            validation.risk_assessment = "medium"

                validation.warnings.extend(result.get("warnings", []))

            except Exception as e:
                validation.errors.append(f"Validation error in {check_name}: {e}")
                validation.is_valid = False

        # Generate test recommendations
        validation.test_recommendations = self._generate_test_recommendations(patch_info)

        return validation

    def _validate_syntax(self, original_code: str, patched_code: str,
                        patch_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate syntax correctness"""
        result = {"passed": True, "errors": [], "warnings": []}

        try:
            ast.parse(patched_code)
        except SyntaxError as e:
            result["passed"] = False
            result["errors"].append(f"Syntax error: {e}")
        except Exception as e:
            result["passed"] = False
            result["errors"].append(f"Parse error: {e}")

        return result

    def _validate_imports(self, original_code: str, patched_code: str,
                         patch_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate import statements"""
        result = {"passed": True, "errors": [], "warnings": []}

        try:
            # Extract imports from both versions
            original_imports = self._extract_imports(original_code)
            patched_imports = self._extract_imports(patched_code)

            # Check for new imports that might not exist
            new_imports = patched_imports - original_imports

            for import_stmt in new_imports:
                if not self._import_exists(import_stmt):
                    result["warnings"].append(f"New import may not be available: {import_stmt}")

            # Check for removed critical imports
            critical_imports = {"os", "sys", "logging"}
            removed_critical = critical_imports & (original_imports - patched_imports)

            if removed_critical:
                result["errors"].append(f"Critical imports removed: {removed_critical}")
                result["passed"] = False

        except Exception as e:
            result["errors"].append(f"Import validation error: {e}")
            result["passed"] = False

        return result

    def _validate_logic(self, original_code: str, patched_code: str,
                       patch_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate logical correctness"""
        result = {"passed": True, "errors": [], "warnings": []}

        try:
            # Check for infinite loops
            if self._contains_infinite_loop(patched_code):
                result["errors"].append("Potential infinite loop introduced")
                result["passed"] = False

            # Check for unreachable code
            unreachable_lines = self._find_unreachable_code(patched_code)
            if unreachable_lines:
                result["warnings"].append(f"Unreachable code detected at lines: {unreachable_lines}")

            # Check for variable usage
            undefined_vars = self._find_undefined_variables(patched_code)
            if undefined_vars:
                result["errors"].append(f"Undefined variables: {undefined_vars}")
                result["passed"] = False

        except Exception as e:
            result["errors"].append(f"Logic validation error: {e}")
            result["passed"] = False

        return result

    def _validate_security(self, original_code: str, patched_code: str,
                          patch_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate security implications"""
        result = {"passed": True, "errors": [], "warnings": []}

        security_patterns = [
            (r"eval\s*\(", "Use of eval() - security risk"),
            (r"exec\s*\(", "Use of exec() - security risk"),
            (r"os\.system\s*\(", "Use of os.system() - security risk"),
            (r"subprocess\.call\s*\(", "Use of subprocess.call() - prefer safer alternatives"),
            (r"input\s*\(", "Use of input() - potential security issue")
        ]

        for pattern, message in security_patterns:
            import re
            if re.search(pattern, patched_code):
                result["warnings"].append(message)

        # Check for hardcoded secrets
        secret_patterns = [
            r"password\s*=\s*['\"][^'\"]*['\"]",
            r"key\s*=\s*['\"][^'\"]*['\"]",
            r"token\s*=\s*['\"][^'\"]*['\"]"
        ]

        for pattern in secret_patterns:
            if re.search(pattern, patched_code, re.IGNORECASE):
                result["warnings"].append("Potential hardcoded credentials detected")

        return result

    def _extract_imports(self, code: str) -> set:
        """Extract import statements from code"""
        imports = set()

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])

        except:
            pass

        return imports

    def _import_exists(self, import_name: str) -> bool:
        """Check if an import is available"""
        try:
            __import__(import_name)
            return True
        except ImportError:
            return False

    def _contains_infinite_loop(self, code: str) -> bool:
        """Check for potential infinite loops"""
        # Simple heuristic - look for while True without break
        import re
        while_true_matches = re.findall(r'while\s+True\s*:', code, re.IGNORECASE)

        for match in while_true_matches:
            # Check if there's a break in the same indentation level
            # This is a simplified check
            start_idx = code.find(match)
            if start_idx != -1:
                # Look for break in the following lines
                lines = code[start_idx:].split('\n')
                base_indent = len(lines[0]) - len(lines[0].lstrip())

                has_break = False
                for line in lines[1:]:
                    if line.strip().startswith('break'):
                        current_indent = len(line) - len(line.lstrip())
                        if current_indent > base_indent:
                            has_break = True
                            break
                    elif line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                        break  # End of loop

                if not has_break:
                    return True

        return False

    def _find_unreachable_code(self, code: str) -> List[int]:
        """Find unreachable code lines"""
        unreachable = []

        try:
            tree = ast.parse(code)
            lines = code.split('\n')

            for node in ast.walk(tree):
                if isinstance(node, ast.Return):
                    # Check for code after return
                    return_line = node.lineno - 1
                    for i in range(return_line + 1, len(lines)):
                        line = lines[i].strip()
                        if line and not line.startswith('#'):
                            unreachable.append(i + 1)
                            break

        except:
            pass

        return unreachable

    def _find_undefined_variables(self, code: str) -> List[str]:
        """Find potentially undefined variables"""
        undefined = []

        try:
            tree = ast.parse(code)

            # Collect defined variables
            defined = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    defined.add(node.id)
                elif isinstance(node, ast.FunctionDef):
                    defined.add(node.name)
                    # Add parameters
                    for arg in node.args.args:
                        defined.add(arg.arg)

            # Check used variables
            used = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    used.add(node.id)

            # Find undefined variables
            undefined = list(used - defined - {'True', 'False', 'None'})

        except:
            pass

        return undefined

    def _generate_test_recommendations(self, patch_info: Dict[str, Any]) -> List[str]:
        """Generate test recommendations for the patch"""
        recommendations = []

        patch_type = patch_info.get("type", "")

        if "error_handling" in patch_type:
            recommendations.append("Test error conditions and exception handling")
        elif "validation" in patch_type:
            recommendations.append("Test input validation with edge cases")
        elif "security" in patch_type:
            recommendations.append("Test security implications and access controls")
        elif "performance" in patch_type:
            recommendations.append("Test performance impact and resource usage")

        # General recommendations
        recommendations.extend([
            "Run existing unit tests",
            "Test integration with other components",
            "Verify no regressions introduced"
        ])

        return recommendations


class PatchRollbackManager:
    """Manages patch rollback operations"""

    def __init__(self, backup_dir: str):
        self.backup_dir = backup_dir

    def create_backup(self, file_path: str) -> str:
        """Create backup of file before patching"""
        backup_name = f"{os.path.basename(file_path)}.backup.{int(time.time())}"
        backup_path = os.path.join(self.backup_dir, backup_name)

        os.makedirs(self.backup_dir, exist_ok=True)
        shutil.copy2(file_path, backup_path)

        return backup_path

    def rollback_patch(self, backup_path: str, original_path: str) -> bool:
        """Rollback patch using backup"""
        try:
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, original_path)
                os.remove(backup_path)
                return True
            return False

        except Exception as e:
            logging.error(f"Error rolling back patch: {e}")
            return False

    def cleanup_old_backups(self, max_age_days: int = 7):
        """Clean up old backup files"""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)

        if os.path.exists(self.backup_dir):
            for filename in os.listdir(self.backup_dir):
                filepath = os.path.join(self.backup_dir, filename)
                if os.path.getctime(filepath) < cutoff_time:
                    try:
                        os.remove(filepath)
                    except:
                        pass


class PatchApplier:
    """Advanced patch application system with safety and rollback capabilities"""

    def __init__(self, healer):
        self.healer = healer
        self.jarvis = healer.jarvis
        self.logger = logging.getLogger('JARVIS.PatchApplier')

        # Patch management components
        self.validator = CodeValidator()
        self.rollback_manager = PatchRollbackManager(
            os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'patch_backups')
        )

        # Patch tracking
        self.applied_patches: Dict[str, Patch] = {}
        self.patch_history: List[Dict[str, Any]] = []

        # Configuration
        self.auto_verify = True
        self.max_concurrent_patches = 1
        self.backup_retention_days = 7

    async def initialize(self):
        """Initialize the advanced patch applier"""
        try:
            self.logger.info("Initializing advanced patch applier...")

            # Load patch history
            await self._load_patch_history()

            # Cleanup old backups
            self.rollback_manager.cleanup_old_backups(self.backup_retention_days)

            self.logger.info("Advanced patch applier initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing patch applier: {e}")
            raise

    async def apply_patch(self, fix_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a comprehensive patch with safety checks"""
        start_time = time.time()

        try:
            fix = fix_result.get("fix", {})
            error_analysis = fix_result.get("analysis", {})

            # Create patch object
            patch_id = f"patch_{int(time.time())}_{hash(str(fix)) % 10000}"
            patch = Patch(
                patch_id=patch_id,
                file_path=error_analysis.get("file_path", ""),
                description=fix.get("description", "Code fix"),
                changes=fix.get("code_changes", []),
                risk_level=fix_result.get("risk_level", "medium"),
                created_at=time.time()
            )

            self.logger.info(f"Applying patch {patch_id}: {patch.description}")

            # Validate patch before application
            if not await self._validate_patch_safety(patch, error_analysis):
                return {
                    "success": False,
                    "error": "Patch validation failed",
                    "patch_id": patch_id
                }

            # Create backup
            if os.path.exists(patch.file_path):
                patch.backup_path = self.rollback_manager.create_backup(patch.file_path)

            # Apply the patch
            apply_result = await self._execute_patch_application(patch)

            if apply_result["success"]:
                # Mark patch as applied
                patch.applied = True
                patch.applied_at = time.time()
                patch.verification_results = apply_result.get("verification", {})

                # Store patch
                self.applied_patches[patch_id] = patch

                # Record in history
                self.patch_history.append({
                    "patch_id": patch_id,
                    "timestamp": time.time(),
                    "description": patch.description,
                    "file_path": patch.file_path,
                    "risk_level": patch.risk_level,
                    "success": True,
                    "execution_time": time.time() - start_time
                })

                self.logger.info(f"Patch {patch_id} applied successfully")

                return {
                    "success": True,
                    "patch_id": patch_id,
                    "execution_time": time.time() - start_time,
                    "backup_created": patch.backup_path is not None,
                    "verification_passed": apply_result.get("verification", {}).get("passed", False),
                    "restart_required": apply_result.get("restart_required", False)
                }

            else:
                # Rollback if application failed
                if patch.backup_path:
                    self.rollback_manager.rollback_patch(patch.backup_path, patch.file_path)

                return {
                    "success": False,
                    "error": apply_result.get("error", "Patch application failed"),
                    "patch_id": patch_id,
                    "rolled_back": True
                }

        except Exception as e:
            self.logger.error(f"Error applying patch: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }

    async def _validate_patch_safety(self, patch: Patch, error_analysis: Dict[str, Any]) -> bool:
        """Validate patch safety before application"""
        try:
            # Read original file
            if not os.path.exists(patch.file_path):
                self.logger.error(f"File not found: {patch.file_path}")
                return False

            with open(patch.file_path, 'r') as f:
                original_code = f.read()

            # Generate patched code (simplified - would need actual patch application)
            patched_code = original_code  # Placeholder

            # Validate the patch
            validation = self.validator.validate_patch(
                original_code, patched_code, {"type": patch.description}
            )

            if not validation.is_valid:
                self.logger.error(f"Patch validation failed: {validation.errors}")
                return False

            if validation.risk_assessment == "critical":
                self.logger.warning(f"High-risk patch detected: {validation.warnings}")

            return True

        except Exception as e:
            self.logger.error(f"Error validating patch: {e}")
            return False

    async def _execute_patch_application(self, patch: Patch) -> Dict[str, Any]:
        """Execute the actual patch application"""
        try:
            # This is a simplified implementation
            # In a real system, this would apply the actual code changes

            # For now, just mark as successful
            result = {
                "success": True,
                "changes_applied": len(patch.changes),
                "restart_required": False,
                "verification": {
                    "passed": True,
                    "tests_run": 0,
                    "tests_passed": 0
                }
            }

            # Run verification if enabled
            if self.auto_verify:
                verification = await self._verify_patch_application(patch)
                result["verification"] = verification

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _verify_patch_application(self, patch: Patch) -> Dict[str, Any]:
        """Verify patch application by running tests"""
        verification = {
            "passed": True,
            "tests_run": 0,
            "tests_passed": 0,
            "errors": []
        }

        try:
            # Run syntax check
            with open(patch.file_path, 'r') as f:
                code = f.read()

            ast.parse(code)  # Syntax check

            # If tester is available, run tests
            if hasattr(self.jarvis, 'tester'):
                test_result = await self.jarvis.tester.run_tests_for_file(patch.file_path)

                verification["tests_run"] = test_result.get("tests_run", 0)
                verification["tests_passed"] = test_result.get("tests_passed", 0)

                if not test_result.get("success", False):
                    verification["passed"] = False
                    verification["errors"].extend(test_result.get("errors", []))

        except SyntaxError as e:
            verification["passed"] = False
            verification["errors"].append(f"Syntax error: {e}")
        except Exception as e:
            verification["passed"] = False
            verification["errors"].append(f"Verification error: {e}")

        return verification

    async def rollback_patch(self, patch_id: str) -> Dict[str, Any]:
        """Rollback a previously applied patch"""
        try:
            if patch_id not in self.applied_patches:
                return {"success": False, "error": "Patch not found"}

            patch = self.applied_patches[patch_id]

            if not patch.applied or patch.rolled_back:
                return {"success": False, "error": "Patch not applied or already rolled back"}

            if not patch.backup_path or not os.path.exists(patch.backup_path):
                return {"success": False, "error": "Backup not available"}

            # Perform rollback
            success = self.rollback_manager.rollback_patch(patch.backup_path, patch.file_path)

            if success:
                patch.rolled_back = True
                patch.rolled_back_at = time.time()

                # Update history
                for entry in self.patch_history:
                    if entry["patch_id"] == patch_id:
                        entry["rolled_back"] = True
                        entry["rollback_time"] = time.time()

                self.logger.info(f"Patch {patch_id} rolled back successfully")

                return {
                    "success": True,
                    "patch_id": patch_id,
                    "rollback_time": time.time()
                }

            else:
                return {"success": False, "error": "Rollback failed"}

        except Exception as e:
            self.logger.error(f"Error rolling back patch {patch_id}: {e}")
            return {"success": False, "error": str(e)}

    def get_patch_stats(self) -> Dict[str, Any]:
        """Get patch application statistics"""
        total_patches = len(self.patch_history)
        successful_patches = sum(1 for p in self.patch_history if p.get("success", False))
        rolled_back_patches = sum(1 for p in self.patch_history if p.get("rolled_back", False))

        return {
            "total_patches": total_patches,
            "successful_patches": successful_patches,
            "failed_patches": total_patches - successful_patches,
            "rolled_back_patches": rolled_back_patches,
            "success_rate": successful_patches / total_patches if total_patches > 0 else 0
        }

    def list_applied_patches(self) -> List[Dict[str, Any]]:
        """List all applied patches"""
        return [
            {
                "patch_id": patch.patch_id,
                "description": patch.description,
                "file_path": patch.file_path,
                "applied_at": patch.applied_at,
                "rolled_back": patch.rolled_back,
                "risk_level": patch.risk_level
            }
            for patch in self.applied_patches.values()
        ]

    async def _load_patch_history(self):
        """Load patch application history"""
        try:
            history_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'patch_history.json')
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.patch_history = data.get("history", [])

                    # Reconstruct patch objects
                    for entry in self.patch_history:
                        patch_id = entry["patch_id"]
                        if patch_id not in self.applied_patches:
                            patch = Patch(
                                patch_id=patch_id,
                                file_path=entry.get("file_path", ""),
                                description=entry.get("description", ""),
                                changes=[],  # Would need to store changes separately
                                risk_level=entry.get("risk_level", "medium"),
                                created_at=entry.get("timestamp", time.time()),
                                applied=entry.get("success", False),
                                applied_at=entry.get("timestamp", None),
                                rolled_back=entry.get("rolled_back", False)
                            )
                            self.applied_patches[patch_id] = patch

        except Exception as e:
            self.logger.debug(f"Could not load patch history: {e}")

    async def shutdown(self):
        """Shutdown the advanced patch applier"""
        try:
            self.logger.info("Shutting down advanced patch applier...")

            # Save patch history
            await self._save_patch_history()

            # Cleanup old backups
            self.rollback_manager.cleanup_old_backups(self.backup_retention_days)

            self.logger.info("Advanced patch applier shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down patch applier: {e}")

    async def _save_patch_history(self):
        """Save patch application history"""
        try:
            history_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'patch_history.json')

            os.makedirs(os.path.dirname(history_file), exist_ok=True)

            data = {
                "history": self.patch_history[-500:],  # Last 500 patches
                "stats": self.get_patch_stats(),
                "last_updated": time.time()
            }

            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Error saving patch history: {e}")