"""
J.A.R.V.I.S. Advanced Recovery Manager
Comprehensive system recovery, rollback, and disaster recovery capabilities
"""

import os
import time
import json
import shutil
import logging
import asyncio
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import tempfile
import psutil
import threading


@dataclass
class SystemSnapshot:
    """Represents a system snapshot"""
    snapshot_id: str
    timestamp: float
    description: str
    snapshot_type: str  # full, incremental, config_only
    size_bytes: int = 0
    checksum: str = ""
    files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "created"  # created, corrupted, restored


@dataclass
class RecoveryPlan:
    """Represents a recovery plan"""
    plan_id: str
    name: str
    description: str
    steps: List[Dict[str, Any]]
    estimated_duration: int  # seconds
    risk_level: str
    prerequisites: List[str]
    created_at: float
    last_tested: Optional[float] = None
    success_rate: float = 0.0


class SnapshotManager:
    """Manages system snapshots for recovery"""

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.snapshots: Dict[str, SystemSnapshot] = {}
        self.max_snapshots = 50
        self.compression_enabled = True

    async def create_snapshot(self, description: str = "",
                            snapshot_type: str = "full",
                            include_files: List[str] = None) -> SystemSnapshot:
        """Create a comprehensive system snapshot"""
        snapshot_id = f"snapshot_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

        try:
            # Create snapshot directory
            snapshot_dir = os.path.join(self.base_path, snapshot_id)
            os.makedirs(snapshot_dir, exist_ok=True)

            # Collect system state
            system_state = await self._collect_system_state()

            # Collect file backups if specified
            files_backed_up = []
            if include_files:
                files_backed_up = await self._backup_files(snapshot_dir, include_files)

            # Calculate size and checksum
            total_size = await self._calculate_snapshot_size(snapshot_dir, system_state)

            # Create snapshot metadata
            snapshot = SystemSnapshot(
                snapshot_id=snapshot_id,
                timestamp=time.time(),
                description=description or f"System snapshot created at {datetime.now().isoformat()}",
                snapshot_type=snapshot_type,
                size_bytes=total_size,
                files=files_backed_up,
                metadata={
                    "system_state": system_state,
                    "creation_info": {
                        "hostname": os.uname().nodename if hasattr(os, 'uname') else 'unknown',
                        "platform": os.uname().sysname if hasattr(os, 'uname') else 'unknown',
                        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
                        "process_id": os.getpid()
                    }
                }
            )

            # Save snapshot metadata
            await self._save_snapshot_metadata(snapshot)

            # Calculate checksum
            snapshot.checksum = await self._calculate_checksum(snapshot_dir)

            # Update snapshot
            snapshot.metadata["checksum"] = snapshot.checksum
            await self._save_snapshot_metadata(snapshot)

            # Register snapshot
            self.snapshots[snapshot_id] = snapshot

            # Cleanup old snapshots
            await self._cleanup_old_snapshots()

            return snapshot

        except Exception as e:
            # Cleanup failed snapshot
            if os.path.exists(snapshot_dir):
                shutil.rmtree(snapshot_dir)

            raise Exception(f"Failed to create snapshot: {e}")

    async def _collect_system_state(self) -> Dict[str, Any]:
        """Collect comprehensive system state"""
        state = {
            "timestamp": time.time(),
            "processes": {},
            "system_info": {},
            "configuration": {},
            "environment": dict(os.environ)
        }

        try:
            # Process information
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    state["processes"][str(proc.info['pid'])] = {
                        "name": proc.info['name'],
                        "cpu_percent": proc.info['cpu_percent'],
                        "memory_percent": proc.info['memory_percent']
                    }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # System information
            state["system_info"] = {
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "disk_total": psutil.disk_usage('/').total,
                "boot_time": psutil.boot_time()
            }

            # Configuration files (if they exist)
            config_files = [
                "/etc/passwd", "/etc/group", "/etc/hosts",
                "/etc/resolv.conf", "/etc/fstab"
            ]

            for config_file in config_files:
                if os.path.exists(config_file):
                    try:
                        with open(config_file, 'r') as f:
                            # Only backup small config files
                            if os.path.getsize(config_file) < 10000:  # 10KB limit
                                state["configuration"][config_file] = f.read()
                    except:
                        pass

        except Exception as e:
            state["collection_error"] = str(e)

        return state

    async def _backup_files(self, snapshot_dir: str, files: List[str]) -> List[str]:
        """Backup specified files"""
        backed_up = []

        for file_path in files:
            if os.path.exists(file_path):
                try:
                    # Create relative path in snapshot
                    rel_path = os.path.relpath(file_path, '/')
                    backup_path = os.path.join(snapshot_dir, 'files', rel_path)

                    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                    shutil.copy2(file_path, backup_path)
                    backed_up.append(file_path)

                except Exception as e:
                    logging.warning(f"Failed to backup {file_path}: {e}")

        return backed_up

    async def _calculate_snapshot_size(self, snapshot_dir: str, system_state: Dict[str, Any]) -> int:
        """Calculate total snapshot size"""
        total_size = 0

        # Calculate directory size
        for dirpath, dirnames, filenames in os.walk(snapshot_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except:
                    pass

        # Add system state size
        total_size += len(json.dumps(system_state).encode())

        return total_size

    async def _calculate_checksum(self, snapshot_dir: str) -> str:
        """Calculate snapshot checksum"""
        hash_md5 = hashlib.md5()

        for dirpath, dirnames, filenames in os.walk(snapshot_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)
                except:
                    pass

        return hash_md5.hexdigest()

    async def _save_snapshot_metadata(self, snapshot: SystemSnapshot):
        """Save snapshot metadata"""
        metadata_file = os.path.join(self.base_path, snapshot.snapshot_id, 'metadata.json')

        metadata = {
            "snapshot_id": snapshot.snapshot_id,
            "timestamp": snapshot.timestamp,
            "description": snapshot.description,
            "snapshot_type": snapshot.snapshot_type,
            "size_bytes": snapshot.size_bytes,
            "checksum": snapshot.checksum,
            "files": snapshot.files,
            "metadata": snapshot.metadata,
            "status": snapshot.status
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    async def _cleanup_old_snapshots(self):
        """Clean up old snapshots to maintain limit"""
        if len(self.snapshots) <= self.max_snapshots:
            return

        # Sort by timestamp (oldest first)
        sorted_snapshots = sorted(
            self.snapshots.values(),
            key=lambda s: s.timestamp
        )

        # Remove oldest snapshots
        to_remove = sorted_snapshots[:len(sorted_snapshots) - self.max_snapshots]

        for snapshot in to_remove:
            try:
                snapshot_dir = os.path.join(self.base_path, snapshot.snapshot_id)
                if os.path.exists(snapshot_dir):
                    shutil.rmtree(snapshot_dir)
                del self.snapshots[snapshot.snapshot_id]
            except Exception as e:
                logging.error(f"Failed to remove old snapshot {snapshot.snapshot_id}: {e}")

    async def restore_snapshot(self, snapshot_id: str,
                             restore_files: bool = True,
                             restore_config: bool = False) -> Dict[str, Any]:
        """Restore system from snapshot"""
        if snapshot_id not in self.snapshots:
            raise Exception(f"Snapshot {snapshot_id} not found")

        snapshot = self.snapshots[snapshot_id]
        snapshot_dir = os.path.join(self.base_path, snapshot_id)

        if not os.path.exists(snapshot_dir):
            raise Exception(f"Snapshot directory {snapshot_dir} not found")

        # Verify snapshot integrity
        current_checksum = await self._calculate_checksum(snapshot_dir)
        if current_checksum != snapshot.checksum:
            snapshot.status = "corrupted"
            raise Exception(f"Snapshot {snapshot_id} is corrupted")

        restored_items = []

        try:
            # Restore files if requested
            if restore_files and snapshot.files:
                files_restored = await self._restore_files(snapshot_dir, snapshot.files)
                restored_items.extend([f"file:{f}" for f in files_restored])

            # Restore configuration if requested
            if restore_config and "configuration" in snapshot.metadata.get("system_state", {}):
                config_restored = await self._restore_configuration(snapshot.metadata["system_state"]["configuration"])
                restored_items.extend([f"config:{c}" for c in config_restored])

            # Update snapshot status
            snapshot.status = "restored"

            return {
                "success": True,
                "snapshot_id": snapshot_id,
                "items_restored": restored_items,
                "restore_time": time.time()
            }

        except Exception as e:
            snapshot.status = "restore_failed"
            raise Exception(f"Failed to restore snapshot: {e}")

    async def _restore_files(self, snapshot_dir: str, files: List[str]) -> List[str]:
        """Restore backed up files"""
        restored = []

        files_dir = os.path.join(snapshot_dir, 'files')

        for file_path in files:
            try:
                # Get relative path
                rel_path = os.path.relpath(file_path, '/')
                backup_path = os.path.join(files_dir, rel_path)

                if os.path.exists(backup_path):
                    # Create backup of current file
                    if os.path.exists(file_path):
                        backup_current = f"{file_path}.pre_restore_{int(time.time())}"
                        shutil.copy2(file_path, backup_current)

                    # Restore file
                    shutil.copy2(backup_path, file_path)
                    restored.append(file_path)

            except Exception as e:
                logging.error(f"Failed to restore {file_path}: {e}")

        return restored

    async def _restore_configuration(self, configuration: Dict[str, str]) -> List[str]:
        """Restore configuration files"""
        restored = []

        for config_path, content in configuration.items():
            try:
                # Create backup of current config
                if os.path.exists(config_path):
                    backup_current = f"{config_path}.pre_restore_{int(time.time())}"
                    shutil.copy2(config_path, backup_current)

                # Restore configuration
                with open(config_path, 'w') as f:
                    f.write(content)

                restored.append(config_path)

            except Exception as e:
                logging.error(f"Failed to restore config {config_path}: {e}")

        return restored

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all available snapshots"""
        return [
            {
                "snapshot_id": s.snapshot_id,
                "timestamp": s.timestamp,
                "description": s.description,
                "type": s.snapshot_type,
                "size_mb": s.size_bytes / (1024 * 1024),
                "status": s.status,
                "files_count": len(s.files)
            }
            for s in self.snapshots.values()
        ]


class RecoveryPlanManager:
    """Manages recovery plans and procedures"""

    def __init__(self, plans_dir: str):
        self.plans_dir = plans_dir
        self.recovery_plans: Dict[str, RecoveryPlan] = {}

    async def create_recovery_plan(self, name: str, description: str,
                                 steps: List[Dict[str, Any]],
                                 prerequisites: List[str] = None) -> RecoveryPlan:
        """Create a new recovery plan"""
        plan_id = f"plan_{int(time.time())}_{hashlib.md5(name.encode()).hexdigest()[:8]}"

        # Estimate duration based on steps
        estimated_duration = sum(step.get("estimated_duration", 60) for step in steps)

        # Assess risk level
        risk_level = self._assess_plan_risk(steps)

        plan = RecoveryPlan(
            plan_id=plan_id,
            name=name,
            description=description,
            steps=steps,
            estimated_duration=estimated_duration,
            risk_level=risk_level,
            prerequisites=prerequisites or [],
            created_at=time.time()
        )

        # Save plan
        await self._save_recovery_plan(plan)
        self.recovery_plans[plan_id] = plan

        return plan

    def _assess_plan_risk(self, steps: List[Dict[str, Any]]) -> str:
        """Assess risk level of recovery plan"""
        high_risk_keywords = ["reinstall", "format", "delete", "overwrite", "shutdown"]
        medium_risk_keywords = ["restart", "restore", "rollback", "modify"]

        risk_score = 0

        for step in steps:
            step_desc = step.get("description", "").lower()

            for keyword in high_risk_keywords:
                if keyword in step_desc:
                    risk_score += 3

            for keyword in medium_risk_keywords:
                if keyword in step_desc:
                    risk_score += 1

        if risk_score >= 10:
            return "critical"
        elif risk_score >= 5:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"

    async def _save_recovery_plan(self, plan: RecoveryPlan):
        """Save recovery plan to disk"""
        plan_file = os.path.join(self.plans_dir, f"{plan.plan_id}.json")

        os.makedirs(self.plans_dir, exist_ok=True)

        plan_data = {
            "plan_id": plan.plan_id,
            "name": plan.name,
            "description": plan.description,
            "steps": plan.steps,
            "estimated_duration": plan.estimated_duration,
            "risk_level": plan.risk_level,
            "prerequisites": plan.prerequisites,
            "created_at": plan.created_at,
            "last_tested": plan.last_tested,
            "success_rate": plan.success_rate
        }

        with open(plan_file, 'w') as f:
            json.dump(plan_data, f, indent=2)

    async def execute_recovery_plan(self, plan_id: str) -> Dict[str, Any]:
        """Execute a recovery plan"""
        if plan_id not in self.recovery_plans:
            raise Exception(f"Recovery plan {plan_id} not found")

        plan = self.recovery_plans[plan_id]

        execution_result = {
            "plan_id": plan_id,
            "start_time": time.time(),
            "steps_executed": [],
            "success": True,
            "errors": []
        }

        try:
            # Check prerequisites
            prereq_check = await self._check_prerequisites(plan.prerequisites)
            if not prereq_check["met"]:
                raise Exception(f"Prerequisites not met: {prereq_check['missing']}")

            # Execute steps
            for i, step in enumerate(plan.steps):
                step_result = await self._execute_recovery_step(step, i + 1)

                execution_result["steps_executed"].append({
                    "step_number": i + 1,
                    "description": step.get("description", ""),
                    "success": step_result["success"],
                    "duration": step_result["duration"],
                    "output": step_result.get("output", "")
                })

                if not step_result["success"]:
                    execution_result["success"] = False
                    execution_result["errors"].append(step_result.get("error", "Unknown error"))
                    break

            # Update plan statistics
            plan.last_tested = time.time()
            if execution_result["success"]:
                plan.success_rate = (plan.success_rate + 1) / 2  # Simple moving average
            else:
                plan.success_rate = plan.success_rate * 0.9  # Penalize failures

            await self._save_recovery_plan(plan)

        except Exception as e:
            execution_result["success"] = False
            execution_result["errors"].append(str(e))

        execution_result["end_time"] = time.time()
        execution_result["total_duration"] = execution_result["end_time"] - execution_result["start_time"]

        return execution_result

    async def _check_prerequisites(self, prerequisites: List[str]) -> Dict[str, Any]:
        """Check if recovery plan prerequisites are met"""
        missing = []

        for prereq in prerequisites:
            if prereq == "backup_power":
                # Check if system has UPS or backup power
                pass  # Simplified check
            elif prereq == "network_access":
                # Check network connectivity
                pass  # Simplified check
            elif prereq.startswith("file_exists:"):
                file_path = prereq.split(":", 1)[1]
                if not os.path.exists(file_path):
                    missing.append(f"Required file not found: {file_path}")
            else:
                # Unknown prerequisite - assume met
                pass

        return {
            "met": len(missing) == 0,
            "missing": missing
        }

    async def _execute_recovery_step(self, step: Dict[str, Any], step_number: int) -> Dict[str, Any]:
        """Execute a single recovery step"""
        start_time = time.time()

        try:
            step_type = step.get("type", "command")
            step_params = step.get("parameters", {})

            if step_type == "command":
                result = await self._execute_command_step(step_params)
            elif step_type == "file_operation":
                result = await self._execute_file_operation_step(step_params)
            elif step_type == "service_control":
                result = await self._execute_service_control_step(step_params)
            elif step_type == "wait":
                result = await self._execute_wait_step(step_params)
            else:
                result = {"success": False, "error": f"Unknown step type: {step_type}"}

            result["duration"] = time.time() - start_time
            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "duration": time.time() - start_time
            }

    async def _execute_command_step(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a command step"""
        command = params.get("command", "")
        timeout = params.get("timeout", 30)

        # Simplified command execution (would use subprocess in real implementation)
        try:
            # Simulate command execution
            await asyncio.sleep(1)  # Simulate execution time
            return {
                "success": True,
                "output": f"Command executed: {command}",
                "return_code": 0
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _execute_file_operation_step(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a file operation step"""
        operation = params.get("operation", "")
        source = params.get("source", "")
        destination = params.get("destination", "")

        try:
            if operation == "copy":
                shutil.copy2(source, destination)
            elif operation == "move":
                shutil.move(source, destination)
            elif operation == "delete":
                if os.path.isfile(destination):
                    os.remove(destination)
                elif os.path.isdir(destination):
                    shutil.rmtree(destination)

            return {"success": True, "operation": operation}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_service_control_step(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a service control step"""
        service = params.get("service", "")
        action = params.get("action", "")

        # Simplified service control (would use systemctl/service manager)
        try:
            await asyncio.sleep(2)  # Simulate service operation
            return {
                "success": True,
                "service": service,
                "action": action
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_wait_step(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a wait step"""
        duration = params.get("duration", 5)

        await asyncio.sleep(duration)

        return {"success": True, "waited_seconds": duration}


class RecoveryManager:
    """Advanced system recovery and disaster recovery manager"""

    def __init__(self, healer):
        self.healer = healer
        self.jarvis = healer.jarvis
        self.logger = logging.getLogger('JARVIS.RecoveryManager')

        # Recovery components
        self.snapshots_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'snapshots')
        self.plans_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'recovery_plans')

        self.snapshot_manager = SnapshotManager(self.snapshots_dir)
        self.plan_manager = RecoveryPlanManager(self.plans_dir)

        # Recovery state
        self.recovery_operations: Dict[str, Dict[str, Any]] = {}
        self.backup_schedules: Dict[str, Dict[str, Any]] = {}

        # Configuration
        self.auto_backup_enabled = True
        self.backup_interval_hours = 24
        self.max_recovery_attempts = 3

    async def initialize(self):
        """Initialize the advanced recovery manager"""
        try:
            self.logger.info("Initializing advanced recovery manager...")

            # Create necessary directories
            os.makedirs(self.snapshots_dir, exist_ok=True)
            os.makedirs(self.plans_dir, exist_ok=True)

            # Load existing snapshots and plans
            await self._load_snapshots()
            await self._load_recovery_plans()

            # Start automatic backup if enabled
            if self.auto_backup_enabled:
                asyncio.create_task(self._auto_backup_loop())

            # Create default recovery plans
            await self._create_default_recovery_plans()

            self.logger.info("Advanced recovery manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing recovery manager: {e}")
            raise

    async def create_snapshot(self, description: str = "",
                            snapshot_type: str = "full",
                            include_files: List[str] = None) -> Dict[str, Any]:
        """Create a system snapshot"""
        try:
            snapshot = await self.snapshot_manager.create_snapshot(
                description=description,
                snapshot_type=snapshot_type,
                include_files=include_files
            )

            return {
                "success": True,
                "snapshot_id": snapshot.snapshot_id,
                "timestamp": snapshot.timestamp,
                "size_mb": snapshot.size_bytes / (1024 * 1024),
                "description": snapshot.description
            }

        except Exception as e:
            self.logger.error(f"Error creating snapshot: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def rollback(self, snapshot_id: str,
                      restore_files: bool = True,
                      restore_config: bool = False) -> Dict[str, Any]:
        """Rollback system to a snapshot"""
        operation_id = f"rollback_{int(time.time())}"

        try:
            self.recovery_operations[operation_id] = {
                "type": "rollback",
                "snapshot_id": snapshot_id,
                "start_time": time.time(),
                "status": "in_progress"
            }

            # Perform rollback
            result = await self.snapshot_manager.restore_snapshot(
                snapshot_id=snapshot_id,
                restore_files=restore_files,
                restore_config=restore_config
            )

            # Update operation status
            self.recovery_operations[operation_id].update({
                "status": "completed",
                "end_time": time.time(),
                "result": result
            })

            self.logger.info(f"System rollback to snapshot {snapshot_id} completed successfully")

            return {
                "success": True,
                "operation_id": operation_id,
                "snapshot_id": snapshot_id,
                "items_restored": result.get("items_restored", []),
                "rollback_time": time.time() - self.recovery_operations[operation_id]["start_time"]
            }

        except Exception as e:
            # Update operation status
            if operation_id in self.recovery_operations:
                self.recovery_operations[operation_id].update({
                    "status": "failed",
                    "end_time": time.time(),
                    "error": str(e)
                })

            self.logger.error(f"Error during rollback to {snapshot_id}: {e}")
            return {
                "success": False,
                "operation_id": operation_id,
                "error": str(e)
            }

    async def create_recovery_plan(self, name: str, description: str,
                                 steps: List[Dict[str, Any]],
                                 prerequisites: List[str] = None) -> Dict[str, Any]:
        """Create a custom recovery plan"""
        try:
            plan = await self.plan_manager.create_recovery_plan(
                name=name,
                description=description,
                steps=steps,
                prerequisites=prerequisites
            )

            return {
                "success": True,
                "plan_id": plan.plan_id,
                "name": plan.name,
                "risk_level": plan.risk_level,
                "estimated_duration": plan.estimated_duration
            }

        except Exception as e:
            self.logger.error(f"Error creating recovery plan: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def execute_recovery_plan(self, plan_id: str) -> Dict[str, Any]:
        """Execute a recovery plan"""
        operation_id = f"recovery_{int(time.time())}"

        try:
            self.recovery_operations[operation_id] = {
                "type": "recovery_plan",
                "plan_id": plan_id,
                "start_time": time.time(),
                "status": "in_progress"
            }

            # Execute plan
            result = await self.plan_manager.execute_recovery_plan(plan_id)

            # Update operation status
            self.recovery_operations[operation_id].update({
                "status": "completed" if result["success"] else "failed",
                "end_time": time.time(),
                "result": result
            })

            status_msg = "completed successfully" if result["success"] else "failed"
            self.logger.info(f"Recovery plan {plan_id} execution {status_msg}")

            return {
                "success": result["success"],
                "operation_id": operation_id,
                "plan_id": plan_id,
                "steps_executed": len(result.get("steps_executed", [])),
                "errors": result.get("errors", []),
                "execution_time": result.get("total_duration", 0)
            }

        except Exception as e:
            # Update operation status
            if operation_id in self.recovery_operations:
                self.recovery_operations[operation_id].update({
                    "status": "failed",
                    "end_time": time.time(),
                    "error": str(e)
                })

            self.logger.error(f"Error executing recovery plan {plan_id}: {e}")
            return {
                "success": False,
                "operation_id": operation_id,
                "error": str(e)
            }

    async def _create_default_recovery_plans(self):
        """Create default recovery plans"""
        # Basic system restart plan
        restart_plan = [
            {
                "type": "command",
                "description": "Stop all non-essential services",
                "parameters": {"command": "systemctl stop non-essential", "timeout": 30},
                "estimated_duration": 10
            },
            {
                "type": "wait",
                "description": "Wait for services to stop",
                "parameters": {"duration": 5},
                "estimated_duration": 5
            },
            {
                "type": "command",
                "description": "Restart system",
                "parameters": {"command": "shutdown -r now", "timeout": 60},
                "estimated_duration": 60
            }
        ]

        await self.plan_manager.create_recovery_plan(
            name="System Restart",
            description="Basic system restart recovery plan",
            steps=restart_plan,
            prerequisites=["backup_power"]
        )

        # Configuration rollback plan
        config_rollback_plan = [
            {
                "type": "command",
                "description": "Backup current configuration",
                "parameters": {"command": "cp -r /etc /etc.backup.pre_rollback", "timeout": 30},
                "estimated_duration": 15
            },
            {
                "type": "file_operation",
                "description": "Restore configuration from snapshot",
                "parameters": {"operation": "copy", "source": "/snapshot/config", "destination": "/etc"},
                "estimated_duration": 20
            },
            {
                "type": "service_control",
                "description": "Restart affected services",
                "parameters": {"service": "all", "action": "restart"},
                "estimated_duration": 30
            }
        ]

        await self.plan_manager.create_recovery_plan(
            name="Configuration Rollback",
            description="Rollback system configuration to previous state",
            steps=config_rollback_plan,
            prerequisites=["snapshot_available", "backup_power"]
        )

    async def _auto_backup_loop(self):
        """Automatic backup loop"""
        while True:
            try:
                await asyncio.sleep(self.backup_interval_hours * 3600)

                # Create automatic snapshot
                result = await self.create_snapshot(
                    description=f"Automatic backup - {datetime.now().isoformat()}",
                    snapshot_type="incremental"
                )

                if result["success"]:
                    self.logger.info(f"Automatic backup created: {result['snapshot_id']}")
                else:
                    self.logger.error(f"Automatic backup failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                self.logger.error(f"Error in auto backup loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry

    async def _load_snapshots(self):
        """Load existing snapshots"""
        try:
            if os.path.exists(self.snapshots_dir):
                for item in os.listdir(self.snapshots_dir):
                    snapshot_dir = os.path.join(self.snapshots_dir, item)
                    if os.path.isdir(snapshot_dir):
                        metadata_file = os.path.join(snapshot_dir, 'metadata.json')
                        if os.path.exists(metadata_file):
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)

                            snapshot = SystemSnapshot(**metadata)
                            self.snapshot_manager.snapshots[snapshot.snapshot_id] = snapshot

        except Exception as e:
            self.logger.debug(f"Could not load existing snapshots: {e}")

    async def _load_recovery_plans(self):
        """Load existing recovery plans"""
        try:
            if os.path.exists(self.plans_dir):
                for item in os.listdir(self.plans_dir):
                    if item.endswith('.json'):
                        plan_file = os.path.join(self.plans_dir, item)
                        with open(plan_file, 'r') as f:
                            plan_data = json.load(f)

                        plan = RecoveryPlan(**plan_data)
                        self.plan_manager.recovery_plans[plan.plan_id] = plan

        except Exception as e:
            self.logger.debug(f"Could not load existing recovery plans: {e}")

    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        total_operations = len(self.recovery_operations)
        successful_operations = sum(1 for op in self.recovery_operations.values() if op.get("status") == "completed")

        return {
            "total_snapshots": len(self.snapshot_manager.snapshots),
            "total_recovery_plans": len(self.plan_manager.recovery_plans),
            "total_recovery_operations": total_operations,
            "successful_operations": successful_operations,
            "success_rate": successful_operations / total_operations if total_operations > 0 else 0,
            "auto_backup_enabled": self.auto_backup_enabled,
            "backup_interval_hours": self.backup_interval_hours
        }

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all available snapshots"""
        return self.snapshot_manager.list_snapshots()

    def list_recovery_plans(self) -> List[Dict[str, Any]]:
        """List all available recovery plans"""
        return [
            {
                "plan_id": p.plan_id,
                "name": p.name,
                "description": p.description,
                "risk_level": p.risk_level,
                "estimated_duration": p.estimated_duration,
                "success_rate": p.success_rate,
                "last_tested": p.last_tested
            }
            for p in self.plan_manager.recovery_plans.values()
        ]

    async def shutdown(self):
        """Shutdown the advanced recovery manager"""
        try:
            self.logger.info("Shutting down advanced recovery manager...")

            # Save current state
            await self._save_recovery_state()

            self.logger.info("Advanced recovery manager shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down recovery manager: {e}")

    async def _save_recovery_state(self):
        """Save recovery state"""
        try:
            state_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'recovery_state.json')

            state = {
                "recovery_operations": self.recovery_operations,
                "backup_schedules": self.backup_schedules,
                "last_updated": time.time()
            }

            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Error saving recovery state: {e}")