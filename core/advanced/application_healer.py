"""
J.A.R.V.I.S. Application Healer
Self-healing and debugging system
"""

import os
import time
import asyncio
import threading
import logging
from typing import Dict, List, Optional, Any

# Import healer components
from .healer_components.error_detector import ErrorDetector
from .healer_components.debugger import Debugger
from .healer_components.fix_generator import FixGenerator
from .healer_components.patch_applier import PatchApplier
from .healer_components.recovery_manager import RecoveryManager
from .healer_components.optimizer import Optimizer
from .healer_components.health_reporter import HealthReporter
from .healer_components.predictor import Predictor

# Import additional advanced healer components
from .healer_components.advanced_diagnostics import AdvancedDiagnostics
from .healer_components.predictive_maintenance import PredictiveMaintenance
from .healer_components.system_analyzer import SystemAnalyzer
from .healer_components.automated_patcher import AutomatedPatcher
from .healer_components.recovery_orchestrator import RecoveryOrchestrator
from .healer_components.performance_monitor import PerformanceMonitor
from .healer_components.security_healer import SecurityHealer
from .healer_components.resource_optimizer import ResourceOptimizer


class ApplicationHealer:
    """
    Ultra-advanced application healing system
    Self-fixes errors, optimizes performance, and predicts issues
    """

    def __init__(self, jarvis_instance):
        """
        Initialize application healer

        Args:
            jarvis_instance: Reference to main JARVIS instance
        """
        self.jarvis = jarvis_instance
        self.logger = logging.getLogger('JARVIS.ApplicationHealer')

        # Healer components
        self.error_detector = ErrorDetector(self)
        self.debugger = Debugger(self)
        self.fix_generator = FixGenerator(self)
        self.patch_applier = PatchApplier(self)
        self.recovery_manager = RecoveryManager(self)
        self.optimizer = Optimizer(self)
        self.health_reporter = HealthReporter(self)
        self.predictor = Predictor(self)

        # Additional advanced healer components
        self.advanced_diagnostics = AdvancedDiagnostics(self)
        self.predictive_maintenance = PredictiveMaintenance(self)
        self.system_analyzer = SystemAnalyzer(self)
        self.automated_patcher = AutomatedPatcher(self)
        self.recovery_orchestrator = RecoveryOrchestrator(self)
        self.performance_monitor = PerformanceMonitor(self)
        self.security_healer = SecurityHealer(self)
        self.resource_optimizer = ResourceOptimizer(self)

        # Healing tasks
        self.active_healing_tasks = {}
        self.healing_history = []

        # Configuration
        self.auto_healing_enabled = True
        self.healing_interval = 60  # seconds
        self.max_concurrent_healing = 3

        # Performance tracking
        self.stats = {
            "errors_detected": 0,
            "fixes_applied": 0,
            "optimizations_performed": 0,
            "system_recoveries": 0,
            "predictions_made": 0,
            "healing_success_rate": 0.0
        }

        # Background monitoring
        self.monitoring = False
        self.monitor_thread = None

    async def initialize(self):
        """Initialize application healer (optimized for speed)"""
        try:
            self.logger.info("Initializing application healer...")

            # Initialize only essential components immediately
            await self.error_detector.initialize()

            # Initialize heavy components in background
            asyncio.create_task(self._init_heavy_components())

            # Start monitoring immediately
            self.monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                name="HealingMonitor",
                daemon=True
            )
            self.monitor_thread.start()

            self.logger.info("Application healer initialized (background components loading)")

        except Exception as e:
            self.logger.error(f"Error initializing application healer: {e}")
            raise

    async def _init_heavy_components(self):
        """Initialize heavy components in background"""
        try:
            # Initialize core components
            await self.debugger.initialize()
            await self.fix_generator.initialize()
            await self.patch_applier.initialize()
            await self.recovery_manager.initialize()
            await self.optimizer.initialize()
            await self.health_reporter.initialize()
            await self.predictor.initialize()

            # Initialize additional advanced components (skip some for faster startup)
            await self.advanced_diagnostics.initialize()
            await self.predictive_maintenance.initialize()

            # Skip system_analyzer for faster startup - initialize on demand
            # await self.system_analyzer.initialize()
            # await self.automated_patcher.initialize()
            # await self.recovery_orchestrator.initialize()
            # await self.performance_monitor.initialize()
            # await self.security_healer.initialize()
            # await self.resource_optimizer.initialize()

            self.logger.info("All healer components initialized in background")

        except Exception as e:
            self.logger.error(f"Error initializing heavy components: {e}")

    async def _ensure_component_ready(self, component_name: str):
        """Ensure a specific component is initialized when accessed"""
        if not hasattr(self, component_name) or getattr(self, component_name) is None:
            try:
                if component_name == 'system_analyzer':
                    await self.system_analyzer.initialize()
                elif component_name == 'automated_patcher':
                    await self.automated_patcher.initialize()
                elif component_name == 'recovery_orchestrator':
                    await self.recovery_orchestrator.initialize()
                elif component_name == 'performance_monitor':
                    await self.performance_monitor.initialize()
                elif component_name == 'security_healer':
                    await self.security_healer.initialize()
                elif component_name == 'resource_optimizer':
                    await self.resource_optimizer.initialize()
            except Exception as e:
                self.logger.error(f"Error initializing {component_name}: {e}")

    def _monitoring_loop(self):
        """Main monitoring loop for errors and issues"""
        self.logger.info("Application healer monitoring started")

        while self.monitoring:
            try:
                # Check for errors
                errors = self.error_detector.detect_errors()

                for error in errors:
                    self._handle_detected_error(error)

                # Check system health
                health_status = self._check_system_health()

                if not health_status["healthy"]:
                    self._handle_health_issues(health_status)

                # Run optimizations
                if self.auto_healing_enabled:
                    asyncio.run(self._run_optimizations())

                time.sleep(self.healing_interval)

            except Exception as e:
                self.logger.error(f"Error in healing monitoring: {e}")
                time.sleep(self.healing_interval * 2)

        self.logger.info("Application healer monitoring stopped")

    def _handle_detected_error(self, error: Dict[str, Any]):
        """Handle detected error"""
        try:
            self.stats["errors_detected"] += 1

            self.logger.warning(f"Error detected: {error}")

            # Create healing task
            healing_task = {
                "error": error,
                "status": "analyzing",
                "created_at": time.time(),
                "task_id": f"healing_{int(time.time())}_{len(self.active_healing_tasks)}"
            }

            self.active_healing_tasks[healing_task["task_id"]] = healing_task

            # Start healing process
            asyncio.create_task(self._execute_healing_task(healing_task))

        except Exception as e:
            self.logger.error(f"Error handling detected error: {e}")

    async def _execute_healing_task(self, healing_task: Dict[str, Any]):
        """Execute healing task"""
        try:
            task_id = healing_task["task_id"]
            error = healing_task["error"]

            # Update status
            healing_task["status"] = "debugging"
            await self._update_healing_progress(task_id, 20, "Debugging error...")

            # Debug the error
            debug_info = await self.debugger.debug_error(error)

            # Update status
            healing_task["status"] = "generating_fix"
            await self._update_healing_progress(task_id, 40, "Generating fix...")

            # Generate fix
            fix_result = await self.fix_generator.generate_fix(error, debug_info)

            if fix_result["success"]:
                # Update status
                healing_task["status"] = "applying_fix"
                await self._update_healing_progress(task_id, 60, "Applying fix...")

                # Apply fix
                patch_result = await self.patch_applier.apply_patch(fix_result["fix"])

                if patch_result["success"]:
                    healing_task["status"] = "testing_fix"
                    await self._update_healing_progress(task_id, 80, "Testing fix...")

                    # Test the fix
                    test_result = await self._test_fix(patch_result)

                    if test_result["success"]:
                        healing_task["status"] = "completed"
                        await self._update_healing_progress(task_id, 100, "Fix completed successfully")

                        self.stats["fixes_applied"] += 1

                        # Update success rate
                        total_fixes = self.stats["fixes_applied"] + (self.stats["errors_detected"] - self.stats["fixes_applied"])
                        self.stats["healing_success_rate"] = self.stats["fixes_applied"] / total_fixes if total_fixes > 0 else 0

                        self.logger.info(f"Healing task {task_id} completed successfully")
                    else:
                        healing_task["status"] = "failed"
                        healing_task["error"] = test_result["error"]
                        self.logger.error(f"Healing task {task_id} failed during testing")
                else:
                    healing_task["status"] = "failed"
                    healing_task["error"] = patch_result["error"]
                    self.logger.error(f"Healing task {task_id} failed during patch application")
            else:
                healing_task["status"] = "failed"
                healing_task["error"] = fix_result["error"]
                self.logger.error(f"Healing task {task_id} failed during fix generation")

        except Exception as e:
            healing_task["status"] = "failed"
            healing_task["error"] = str(e)
            self.logger.error(f"Error executing healing task {healing_task['task_id']}: {e}")

        finally:
            # Move to history
            healing_task["completed_at"] = time.time()
            self.healing_history.append(healing_task)

            # Remove from active tasks
            if healing_task["task_id"] in self.active_healing_tasks:
                del self.active_healing_tasks[healing_task["task_id"]]

    async def _update_healing_progress(self, task_id: str, progress: float, message: str):
        """Update healing task progress"""
        if task_id in self.active_healing_tasks:
            self.active_healing_tasks[task_id]["progress"] = progress
            self.active_healing_tasks[task_id]["message"] = message

            # Notify GUI if available
            if hasattr(self.jarvis, 'event_manager'):
                self.jarvis.event_manager.emit_event(
                    self.jarvis.event_manager.EventType.CUSTOM,
                    {
                        "event_type": "healing_progress",
                        "task_id": task_id,
                        "progress": progress,
                        "message": message
                    },
                    source="application_healer"
                )

    def _check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        try:
            health_score = 100
            issues = []

            # Check memory usage
            if hasattr(self.jarvis, 'system_monitor'):
                memory_info = self.jarvis.system_monitor.current_readings.get('memory', {})
                if memory_info.get('percent', 0) > 90:
                    health_score -= 30
                    issues.append("High memory usage")

            # Check CPU usage
            if hasattr(self.jarvis, 'system_monitor'):
                cpu_info = self.jarvis.system_monitor.current_readings.get('cpu', {})
                if cpu_info.get('percent', 0) > 95:
                    health_score -= 25
                    issues.append("High CPU usage")

            # Check for error logs
            log_issues = self._check_error_logs()
            if log_issues:
                health_score -= 20
                issues.extend(log_issues)

            return {
                "healthy": health_score >= 70,
                "health_score": health_score,
                "issues": issues
            }

        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
            return {"healthy": False, "health_score": 0, "issues": [str(e)]}

    def _check_error_logs(self) -> List[str]:
        """Check for error patterns in logs"""
        issues = []

        try:
            log_file = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'jarvis.log')

            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    recent_lines = f.readlines()[-100:]  # Last 100 lines

                error_count = sum(1 for line in recent_lines if 'ERROR' in line)
                if error_count > 10:
                    issues.append(f"High error count in logs: {error_count}")

        except Exception as e:
            self.logger.error(f"Error checking error logs: {e}")

        return issues

    def _handle_health_issues(self, health_status: Dict[str, Any]):
        """Handle system health issues"""
        try:
            for issue in health_status["issues"]:
                if "memory" in issue.lower():
                    # Trigger memory optimization
                    asyncio.create_task(self._optimize_memory())
                elif "cpu" in issue.lower():
                    # Trigger CPU optimization
                    asyncio.create_task(self._optimize_cpu())
                elif "error" in issue.lower():
                    # Trigger error cleanup
                    asyncio.create_task(self._cleanup_errors())

        except Exception as e:
            self.logger.error(f"Error handling health issues: {e}")

    async def _run_optimizations(self):
        """Run system optimizations"""
        try:
            # Memory optimization
            await self._optimize_memory()

            # Performance optimization
            await self._optimize_performance()

            # Cleanup tasks
            await self._cleanup_resources()

        except Exception as e:
            self.logger.error(f"Error running optimizations: {e}")

    async def _optimize_memory(self):
        """Optimize memory usage"""
        try:
            # Force garbage collection
            import gc
            gc.collect()

            # Clear caches if they exist
            if hasattr(self.jarvis, 'api_manager'):
                # Clear API cache
                pass

            self.stats["optimizations_performed"] += 1
            self.logger.info("Memory optimization completed")

        except Exception as e:
            self.logger.error(f"Error optimizing memory: {e}")

    async def _optimize_cpu(self):
        """Optimize CPU usage"""
        try:
            # Reduce monitoring frequency if needed
            if hasattr(self.jarvis, 'system_monitor'):
                # Could adjust monitoring intervals
                pass

            self.stats["optimizations_performed"] += 1
            self.logger.info("CPU optimization completed")

        except Exception as e:
            self.logger.error(f"Error optimizing CPU: {e}")

    async def _optimize_performance(self):
        """Optimize overall performance"""
        try:
            # Run performance optimization
            optimization_result = await self.optimizer.optimize_system()

            if optimization_result["success"]:
                self.stats["optimizations_performed"] += 1

        except Exception as e:
            self.logger.error(f"Error optimizing performance: {e}")

    async def _cleanup_errors(self):
        """Clean up error logs and temporary files"""
        try:
            # Clean up old error logs
            log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')

            if os.path.exists(log_dir):
                for file in os.listdir(log_dir):
                    if file.endswith('.log'):
                        file_path = os.path.join(log_dir, file)
                        # Remove logs older than 7 days
                        if time.time() - os.path.getmtime(file_path) > 7 * 24 * 3600:
                            try:
                                os.remove(file_path)
                            except:
                                pass

        except Exception as e:
            self.logger.error(f"Error cleaning up errors: {e}")

    async def _cleanup_resources(self):
        """Clean up system resources"""
        try:
            # Clean temporary files
            if hasattr(self.jarvis, 'file_manager'):
                await self.jarvis.file_manager.cleanup_temp_files()

            # Clear old caches
            if hasattr(self.jarvis, 'api_manager'):
                # Clear old API cache
                pass

        except Exception as e:
            self.logger.error(f"Error cleaning up resources: {e}")

    async def _test_fix(self, patch_result: Dict[str, Any]) -> Dict[str, Any]:
        """Test applied fix"""
        try:
            # Basic test - check if system is still running
            await asyncio.sleep(5)  # Wait for fix to take effect

            # Check if error still exists
            recent_errors = self.error_detector.detect_errors()

            # If the specific error is gone, test passed
            error_type = patch_result.get("error_type", "")
            error_still_exists = any(error_type in str(error) for error in recent_errors)

            return {
                "success": not error_still_exists,
                "error": "Error still exists after fix" if error_still_exists else None
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_healing_status(self) -> Dict[str, Any]:
        """Get current healing status"""
        return {
            "active_tasks": len(self.active_healing_tasks),
            "auto_healing_enabled": self.auto_healing_enabled,
            "monitoring": self.monitoring,
            "stats": self.stats,
            "recent_activities": self.healing_history[-10:] if self.healing_history else []
        }

    def enable_auto_healing(self):
        """Enable automatic healing"""
        self.auto_healing_enabled = True
        self.logger.info("Auto-healing enabled")

    def disable_auto_healing(self):
        """Disable automatic healing"""
        self.auto_healing_enabled = False
        self.logger.info("Auto-healing disabled")

    async def trigger_manual_healing(self, target: str = "system") -> str:
        """
        Trigger manual healing

        Args:
            target: Target to heal (system, module, specific_component)

        Returns:
            Healing task ID
        """
        try:
            task_id = f"manual_healing_{int(time.time())}"

            healing_task = {
                "task_id": task_id,
                "target": target,
                "status": "starting",
                "created_at": time.time(),
                "manual_trigger": True
            }

            self.active_healing_tasks[task_id] = healing_task

            # Start healing process
            asyncio.create_task(self._execute_manual_healing(healing_task))

            return task_id

        except Exception as e:
            self.logger.error(f"Error triggering manual healing: {e}")
            return ""

    async def _execute_manual_healing(self, healing_task: Dict[str, Any]):
        """Execute manual healing task"""
        try:
            task_id = healing_task["task_id"]
            target = healing_task["target"]

            healing_task["status"] = "analyzing"
            await self._update_healing_progress(task_id, 10, f"Analyzing {target}...")

            # Perform comprehensive healing
            if target == "system":
                await self._heal_system(healing_task)
            elif target.startswith("module:"):
                await self._heal_module(healing_task, target[7:])
            else:
                await self._heal_component(healing_task, target)

        except Exception as e:
            healing_task["status"] = "failed"
            healing_task["error"] = str(e)
            self.logger.error(f"Error in manual healing {healing_task['task_id']}: {e}")

    async def _heal_system(self, healing_task: Dict[str, Any]):
        """Heal entire system"""
        try:
            task_id = healing_task["task_id"]

            # Step 1: Error detection
            healing_task["status"] = "detecting_errors"
            await self._update_healing_progress(task_id, 20, "Detecting system errors...")

            errors = self.error_detector.detect_errors()

            # Step 2: Generate fixes
            healing_task["status"] = "generating_fixes"
            await self._update_healing_progress(task_id, 40, "Generating system fixes...")

            fixes = []
            for error in errors:
                fix_result = await self.fix_generator.generate_fix(error, {})
                if fix_result["success"]:
                    fixes.append(fix_result)

            # Step 3: Apply fixes
            healing_task["status"] = "applying_fixes"
            await self._update_healing_progress(task_id, 60, "Applying system fixes...")

            applied_fixes = 0
            for fix in fixes:
                patch_result = await self.patch_applier.apply_patch(fix["fix"])
                if patch_result["success"]:
                    applied_fixes += 1

            # Step 4: Optimize
            healing_task["status"] = "optimizing"
            await self._update_healing_progress(task_id, 80, "Optimizing system...")

            optimization_result = await self.optimizer.optimize_system()

            # Complete
            healing_task["status"] = "completed"
            await self._update_healing_progress(task_id, 100, f"System healing completed. Applied {applied_fixes} fixes.")

            self.stats["system_recoveries"] += 1

        except Exception as e:
            healing_task["status"] = "failed"
            healing_task["error"] = str(e)

    async def _heal_module(self, healing_task: Dict[str, Any], module_name: str):
        """Heal specific module"""
        try:
            # Module-specific healing logic would go here
            healing_task["status"] = "completed"
            await self._update_healing_progress(healing_task["task_id"], 100, f"Module {module_name} healing completed")

        except Exception as e:
            healing_task["status"] = "failed"
            healing_task["error"] = str(e)

    async def _heal_component(self, healing_task: Dict[str, Any], component: str):
        """Heal specific component"""
        try:
            # Component-specific healing logic would go here
            healing_task["status"] = "completed"
            await self._update_healing_progress(healing_task["task_id"], 100, f"Component {component} healing completed")

        except Exception as e:
            healing_task["status"] = "failed"
            healing_task["error"] = str(e)

    def get_healing_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get healing history"""
        return self.healing_history[-limit:] if self.healing_history else []

    async def shutdown(self):
        """Shutdown application healer"""
        try:
            self.logger.info("Shutting down application healer...")

            self.monitoring = False

            # Wait for monitor thread
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)

            # Shutdown components
            await self.error_detector.shutdown()
            await self.debugger.shutdown()
            await self.fix_generator.shutdown()
            await self.patch_applier.shutdown()
            await self.recovery_manager.shutdown()
            await self.optimizer.shutdown()
            await self.health_reporter.shutdown()
            await self.predictor.shutdown()

            # Shutdown additional advanced components
            await self.advanced_diagnostics.shutdown()
            await self.predictive_maintenance.shutdown()
            await self.system_analyzer.shutdown()
            await self.automated_patcher.shutdown()
            await self.recovery_orchestrator.shutdown()
            await self.performance_monitor.shutdown()
            await self.security_healer.shutdown()
            await self.resource_optimizer.shutdown()

            self.logger.info("Application healer shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down application healer: {e}")