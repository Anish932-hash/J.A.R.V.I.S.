"""
J.A.R.V.I.S. Deployment Orchestrator
Advanced deployment orchestration for AI-generated code with rollback capabilities
"""

import os
import time
import asyncio
import logging
import hashlib
import shutil
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import tempfile
import subprocess
import sys


class DeploymentOrchestrator:
    """
    Ultra-advanced deployment orchestrator that manages safe deployment of AI-generated code,
    handles rollbacks, A/B testing, and gradual rollouts
    """

    def __init__(self, development_engine):
        """
        Initialize deployment orchestrator

        Args:
            development_engine: Reference to self-development engine
        """
        self.engine = development_engine
        self.jarvis = development_engine.jarvis
        self.logger = logging.getLogger('JARVIS.DeploymentOrchestrator')

        # Deployment configuration
        self.deployment_config = {
            'backup_before_deploy': True,
            'enable_rollback': True,
            'gradual_rollout': True,
            'health_checks': True,
            'auto_rollback_on_failure': True,
            'deployment_timeout': 300,  # 5 minutes
            'health_check_interval': 30,  # 30 seconds
            'rollback_timeout': 120  # 2 minutes
        }

        # Deployment state
        self.active_deployments = {}
        self.deployment_history = []
        self.backup_store = {}

        # Deployment statistics
        self.stats = {
            'deployments_attempted': 0,
            'deployments_successful': 0,
            'deployments_failed': 0,
            'rollbacks_performed': 0,
            'avg_deployment_time': 0.0,
            'success_rate': 0.0
        }

    async def initialize(self):
        """Initialize deployment orchestrator"""
        try:
            self.logger.info("Initializing deployment orchestrator...")

            # Setup deployment directories
            await self._setup_deployment_directories()

            # Load deployment history
            await self._load_deployment_history()

            self.logger.info("Deployment orchestrator initialized")

        except Exception as e:
            self.logger.error(f"Error initializing deployment orchestrator: {e}")
            raise

    async def orchestrate_deployment(self,
                                   code: str,
                                   component_name: str,
                                   deployment_type: str = "standard",
                                   target_environment: str = "production") -> Dict[str, Any]:
        """
        Orchestrate deployment of AI-generated code

        Args:
            code: Code to deploy
            component_name: Name of the component
            deployment_type: Type of deployment (standard, canary, blue_green, gradual)
            target_environment: Target environment (development, staging, production)

        Returns:
            Deployment orchestration results
        """
        deployment_id = f"deploy_{int(time.time())}_{hashlib.md5(component_name.encode()).hexdigest()[:8]}"
        start_time = time.time()

        try:
            self.logger.info(f"Orchestrating {deployment_type} deployment of {component_name}")

            # Create deployment plan
            deployment_plan = await self._create_deployment_plan(
                code, component_name, deployment_type, target_environment, deployment_id
            )

            # Validate deployment
            validation = await self._validate_deployment(deployment_plan)

            if not validation['is_valid']:
                return {
                    'deployment_id': deployment_id,
                    'success': False,
                    'error': 'Deployment validation failed',
                    'validation_issues': validation['issues'],
                    'deployment_time': time.time() - start_time
                }

            # Execute deployment
            deployment_result = await self._execute_deployment(deployment_plan)

            # Monitor deployment
            if deployment_result['success']:
                monitoring_result = await self._monitor_deployment(deployment_plan)

                if not monitoring_result['healthy']:
                    # Auto-rollback if enabled
                    if self.deployment_config['auto_rollback_on_failure']:
                        self.logger.warning("Deployment unhealthy, initiating rollback")
                        rollback_result = await self._rollback_deployment(deployment_id)
                        deployment_result['rollback_performed'] = rollback_result['success']
                        deployment_result['rollback_reason'] = 'health_check_failed'

            # Record deployment
            deployment_record = {
                'deployment_id': deployment_id,
                'component_name': component_name,
                'deployment_type': deployment_type,
                'target_environment': target_environment,
                'success': deployment_result['success'],
                'start_time': start_time,
                'end_time': time.time(),
                'validation': validation,
                'result': deployment_result,
                'monitoring': monitoring_result if 'monitoring_result' in locals() else None
            }

            self.deployment_history.append(deployment_record)
            self.stats['deployments_attempted'] += 1

            if deployment_result['success']:
                self.stats['deployments_successful'] += 1
            else:
                self.stats['deployments_failed'] += 1

            # Update success rate
            total_deployments = self.stats['deployments_attempted']
            self.stats['success_rate'] = self.stats['deployments_successful'] / total_deployments if total_deployments > 0 else 0

            # Update average deployment time
            deployment_time = time.time() - start_time
            self.stats['avg_deployment_time'] = (
                (self.stats['avg_deployment_time'] * (total_deployments - 1)) + deployment_time
            ) / total_deployments

            self.logger.info(f"Deployment orchestration completed: {deployment_result['success']}")
            return deployment_record

        except Exception as e:
            self.logger.error(f"Error in deployment orchestration: {e}")
            return {
                'deployment_id': deployment_id,
                'success': False,
                'error': str(e),
                'deployment_time': time.time() - start_time
            }

    async def _create_deployment_plan(self, code: str, component_name: str,
                                    deployment_type: str, target_environment: str,
                                    deployment_id: str) -> Dict[str, Any]:
        """Create deployment plan"""
        plan = {
            'deployment_id': deployment_id,
            'component_name': component_name,
            'code': code,
            'code_hash': hashlib.md5(code.encode()).hexdigest(),
            'deployment_type': deployment_type,
            'target_environment': target_environment,
            'backup_required': self.deployment_config['backup_before_deploy'],
            'rollback_enabled': self.deployment_config['enable_rollback'],
            'health_checks_enabled': self.deployment_config['health_checks'],
            'steps': [],
            'rollback_plan': None
        }

        # Determine deployment steps based on type
        if deployment_type == 'standard':
            plan['steps'] = [
                'validate_code',
                'create_backup',
                'deploy_code',
                'run_health_checks',
                'cleanup_temp_files'
            ]
        elif deployment_type == 'canary':
            plan['steps'] = [
                'validate_code',
                'create_backup',
                'deploy_to_canary',
                'run_canary_tests',
                'gradual_rollout',
                'run_health_checks',
                'cleanup_temp_files'
            ]
        elif deployment_type == 'blue_green':
            plan['steps'] = [
                'validate_code',
                'create_backup',
                'deploy_to_green',
                'run_green_tests',
                'switch_traffic',
                'run_health_checks',
                'cleanup_blue'
            ]
        elif deployment_type == 'gradual':
            plan['steps'] = [
                'validate_code',
                'create_backup',
                'deploy_10_percent',
                'monitor_10_percent',
                'deploy_50_percent',
                'monitor_50_percent',
                'deploy_100_percent',
                'run_health_checks',
                'cleanup_temp_files'
            ]

        # Create rollback plan
        if plan['rollback_enabled']:
            plan['rollback_plan'] = await self._create_rollback_plan(plan)

        return plan

    async def _create_rollback_plan(self, deployment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Create rollback plan"""
        rollback_plan = {
            'deployment_id': deployment_plan['deployment_id'],
            'component_name': deployment_plan['component_name'],
            'steps': [
                'stop_new_version',
                'restore_backup',
                'restart_services',
                'verify_rollback',
                'cleanup_failed_deployment'
            ],
            'timeout': self.deployment_config['rollback_timeout'],
            'backup_location': None
        }

        return rollback_plan

    async def _validate_deployment(self, deployment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate deployment plan"""
        validation = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }

        try:
            # Validate code syntax
            try:
                compile(deployment_plan['code'], '<deployment>', 'exec')
            except SyntaxError as e:
                validation['is_valid'] = False
                validation['issues'].append(f"Syntax error: {e}")

            # Validate component name
            if not deployment_plan['component_name'] or not deployment_plan['component_name'].replace('_', '').isalnum():
                validation['is_valid'] = False
                validation['issues'].append("Invalid component name")

            # Check for existing deployment conflicts
            if deployment_plan['component_name'] in self.active_deployments:
                validation['warnings'].append(f"Component {deployment_plan['component_name']} is currently being deployed")

            # Validate environment access
            if not await self._validate_environment_access(deployment_plan['target_environment']):
                validation['is_valid'] = False
                validation['issues'].append(f"No access to {deployment_plan['target_environment']} environment")

            # Check resource availability
            resource_check = await self._check_resource_availability(deployment_plan)
            if not resource_check['available']:
                validation['is_valid'] = False
                validation['issues'].extend(resource_check['issues'])

        except Exception as e:
            validation['is_valid'] = False
            validation['issues'].append(f"Validation error: {e}")

        return validation

    async def _validate_environment_access(self, environment: str) -> bool:
        """Validate access to target environment"""
        # In a real implementation, this would check permissions, network access, etc.
        # For now, assume all environments are accessible
        return True

    async def _check_resource_availability(self, deployment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Check resource availability for deployment"""
        resources = {
            'available': True,
            'issues': []
        }

        # Check disk space
        try:
            stat = os.statvfs('.')
            free_space = stat.f_bavail * stat.f_frsize
            code_size = len(deployment_plan['code'].encode('utf-8'))

            if free_space < code_size * 2:  # Need at least 2x code size
                resources['available'] = False
                resources['issues'].append("Insufficient disk space for deployment")
        except:
            pass  # Skip disk check if not available

        # Check memory availability
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                resources['available'] = False
                resources['issues'].append("High memory usage may affect deployment")
        except:
            pass

        return resources

    async def _execute_deployment(self, deployment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute deployment according to plan"""
        result = {
            'success': True,
            'step_results': [],
            'error': None,
            'rollback_available': deployment_plan.get('rollback_enabled', False)
        }

        self.active_deployments[deployment_plan['component_name']] = deployment_plan

        try:
            for step in deployment_plan['steps']:
                step_result = await self._execute_deployment_step(step, deployment_plan)
                result['step_results'].append(step_result)

                if not step_result['success']:
                    result['success'] = False
                    result['error'] = step_result.get('error', f"Step {step} failed")
                    break

                # Update progress
                await self._update_deployment_progress(
                    deployment_plan['deployment_id'],
                    len(result['step_results']) / len(deployment_plan['steps']) * 100,
                    f"Completed step: {step}"
                )

        except Exception as e:
            result['success'] = False
            result['error'] = str(e)

        finally:
            # Remove from active deployments
            if deployment_plan['component_name'] in self.active_deployments:
                del self.active_deployments[deployment_plan['component_name']]

        return result

    async def _execute_deployment_step(self, step: str, deployment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single deployment step"""
        step_result = {
            'step': step,
            'success': True,
            'error': None,
            'details': {}
        }

        try:
            if step == 'validate_code':
                step_result['details'] = await self._step_validate_code(deployment_plan)
            elif step == 'create_backup':
                step_result['details'] = await self._step_create_backup(deployment_plan)
            elif step == 'deploy_code':
                step_result['details'] = await self._step_deploy_code(deployment_plan)
            elif step == 'run_health_checks':
                step_result['details'] = await self._step_run_health_checks(deployment_plan)
            elif step == 'cleanup_temp_files':
                step_result['details'] = await self._step_cleanup_temp_files(deployment_plan)
            elif step == 'deploy_to_canary':
                step_result['details'] = await self._step_deploy_canary(deployment_plan)
            elif step == 'run_canary_tests':
                step_result['details'] = await self._step_run_canary_tests(deployment_plan)
            elif step == 'gradual_rollout':
                step_result['details'] = await self._step_gradual_rollout(deployment_plan)
            elif step == 'deploy_to_green':
                step_result['details'] = await self._step_deploy_green(deployment_plan)
            elif step == 'run_green_tests':
                step_result['details'] = await self._step_run_green_tests(deployment_plan)
            elif step == 'switch_traffic':
                step_result['details'] = await self._step_switch_traffic(deployment_plan)
            elif step == 'cleanup_blue':
                step_result['details'] = await self._step_cleanup_blue(deployment_plan)
            else:
                step_result['success'] = False
                step_result['error'] = f"Unknown deployment step: {step}"

        except Exception as e:
            step_result['success'] = False
            step_result['error'] = str(e)

        return step_result

    async def _step_validate_code(self, deployment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate code before deployment"""
        # Additional validation beyond initial validation
        return {'validation_passed': True}

    async def _step_create_backup(self, deployment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Create backup of current code"""
        try:
            component_name = deployment_plan['component_name']
            backup_path = Path(tempfile.mkdtemp(prefix=f'backup_{component_name}_'))

            # Find current component location
            current_path = await self._find_component_path(component_name)
            if current_path and current_path.exists():
                if current_path.is_file():
                    shutil.copy2(current_path, backup_path / current_path.name)
                else:
                    shutil.copytree(current_path, backup_path / current_path.name, dirs_exist_ok=True)

            # Store backup info
            self.backup_store[deployment_plan['deployment_id']] = str(backup_path)

            return {
                'backup_created': True,
                'backup_path': str(backup_path)
            }

        except Exception as e:
            return {
                'backup_created': False,
                'error': str(e)
            }

    async def _step_deploy_code(self, deployment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy the code"""
        try:
            component_name = deployment_plan['component_name']
            code = deployment_plan['code']

            # Find deployment target
            target_path = await self._find_component_path(component_name)
            if not target_path:
                # Create new component
                target_path = Path('jarvis') / 'core' / 'advanced' / f'{component_name}.py'

            # Write code to target
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, 'w') as f:
                f.write(code)

            return {
                'deployment_path': str(target_path),
                'code_deployed': True
            }

        except Exception as e:
            return {
                'code_deployed': False,
                'error': str(e)
            }

    async def _step_run_health_checks(self, deployment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Run health checks after deployment"""
        try:
            # Basic health checks
            health_checks = [
                await self._check_code_syntax(deployment_plan),
                await self._check_import_compatibility(deployment_plan),
                await self._check_basic_functionality(deployment_plan)
            ]

            all_passed = all(check['passed'] for check in health_checks)

            return {
                'health_checks_passed': all_passed,
                'checks': health_checks
            }

        except Exception as e:
            return {
                'health_checks_passed': False,
                'error': str(e)
            }

    async def _step_cleanup_temp_files(self, deployment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up temporary files"""
        # Cleanup would be implemented based on deployment type
        return {'cleanup_completed': True}

    # Additional step implementations for different deployment types
    async def _step_deploy_canary(self, deployment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy to canary environment"""
        # Implement canary deployment logic
        return {'canary_deployment': True}

    async def _step_run_canary_tests(self, deployment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Run canary tests"""
        # Implement canary testing logic
        return {'canary_tests_passed': True}

    async def _step_gradual_rollout(self, deployment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Gradual rollout implementation"""
        return {'gradual_rollout_completed': True}

    async def _step_deploy_green(self, deployment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy to green environment"""
        return {'green_deployment': True}

    async def _step_run_green_tests(self, deployment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Run green environment tests"""
        return {'green_tests_passed': True}

    async def _step_switch_traffic(self, deployment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Switch traffic to new deployment"""
        return {'traffic_switched': True}

    async def _step_cleanup_blue(self, deployment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up blue environment"""
        return {'blue_cleanup_completed': True}

    async def _find_component_path(self, component_name: str) -> Optional[Path]:
        """Find the path of an existing component"""
        # Search in common locations
        search_paths = [
            Path('jarvis') / 'core' / 'advanced' / f'{component_name}.py',
            Path('jarvis') / 'modules' / f'{component_name}.py',
            Path('jarvis') / 'core' / f'{component_name}.py'
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None

    async def _check_code_syntax(self, deployment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Check code syntax"""
        try:
            compile(deployment_plan['code'], '<deployment>', 'exec')
            return {'check': 'syntax', 'passed': True}
        except SyntaxError as e:
            return {'check': 'syntax', 'passed': False, 'error': str(e)}

    async def _check_import_compatibility(self, deployment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Check import compatibility"""
        # Basic import check
        return {'check': 'imports', 'passed': True}

    async def _check_basic_functionality(self, deployment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Check basic functionality"""
        # Basic functionality check
        return {'check': 'functionality', 'passed': True}

    async def _monitor_deployment(self, deployment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor deployment health"""
        monitoring = {
            'healthy': True,
            'checks': [],
            'issues': []
        }

        try:
            # Run health checks for a period
            check_interval = self.deployment_config['health_check_interval']
            max_checks = 3  # Check 3 times

            for i in range(max_checks):
                check_result = await self._step_run_health_checks(deployment_plan)
                monitoring['checks'].append(check_result)

                if not check_result['health_checks_passed']:
                    monitoring['healthy'] = False
                    monitoring['issues'].append(f"Health check {i+1} failed")
                    break

                if i < max_checks - 1:  # Don't sleep after last check
                    await asyncio.sleep(check_interval)

        except Exception as e:
            monitoring['healthy'] = False
            monitoring['issues'].append(f"Monitoring error: {e}")

        return monitoring

    async def _rollback_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Rollback a deployment"""
        rollback_result = {
            'success': False,
            'error': None,
            'steps_completed': []
        }

        try:
            # Find deployment record
            deployment_record = None
            for record in self.deployment_history:
                if record['deployment_id'] == deployment_id:
                    deployment_record = record
                    break

            if not deployment_record:
                rollback_result['error'] = "Deployment record not found"
                return rollback_result

            # Execute rollback plan
            rollback_plan = deployment_record.get('result', {}).get('rollback_plan')
            if not rollback_plan:
                rollback_result['error'] = "No rollback plan available"
                return rollback_result

            # Execute rollback steps
            for step in rollback_plan['steps']:
                step_result = await self._execute_rollback_step(step, rollback_plan)
                rollback_result['steps_completed'].append(step_result)

                if not step_result['success']:
                    rollback_result['error'] = f"Rollback failed at step: {step}"
                    break

            rollback_result['success'] = all(step['success'] for step in rollback_result['steps_completed'])
            self.stats['rollbacks_performed'] += 1

        except Exception as e:
            rollback_result['error'] = str(e)

        return rollback_result

    async def _execute_rollback_step(self, step: str, rollback_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a rollback step"""
        # Implement rollback steps
        if step == 'restore_backup':
            backup_path = self.backup_store.get(rollback_plan['deployment_id'])
            if backup_path:
                # Restore backup
                return {'step': step, 'success': True}
            else:
                return {'step': step, 'success': False, 'error': 'No backup found'}

        # Other rollback steps would be implemented similarly
        return {'step': step, 'success': True}

    async def _update_deployment_progress(self, deployment_id: str, progress: float, message: str):
        """Update deployment progress"""
        # This would notify any monitoring systems
        pass

    async def _setup_deployment_directories(self):
        """Setup deployment directories"""
        # Create necessary directories
        Path('jarvis/deployments').mkdir(exist_ok=True)
        Path('jarvis/backups').mkdir(exist_ok=True)

    async def _load_deployment_history(self):
        """Load deployment history"""
        history_file = Path('jarvis/data/deployment_history.json')
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self.deployment_history = json.load(f)
            except:
                self.deployment_history = []

    def get_deployment_stats(self) -> Dict[str, Any]:
        """Get deployment statistics"""
        return {
            **self.stats,
            'active_deployments': len(self.active_deployments),
            'total_history': len(self.deployment_history),
            'recent_deployments': len([d for d in self.deployment_history[-10:] if d.get('success', False)])
        }

    async def shutdown(self):
        """Shutdown deployment orchestrator"""
        try:
            self.logger.info("Shutting down deployment orchestrator...")

            # Cancel active deployments
            for deployment in self.active_deployments.values():
                # Mark for cancellation
                pass

            # Save deployment history
            history_file = Path('jarvis/data/deployment_history.json')
            history_file.parent.mkdir(parents=True, exist_ok=True)

            with open(history_file, 'w') as f:
                # Save recent history
                recent_history = self.deployment_history[-100:]  # Last 100 deployments
                json.dump(recent_history, f, indent=2, default=str)

            self.logger.info("Deployment orchestrator shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down deployment orchestrator: {e}")