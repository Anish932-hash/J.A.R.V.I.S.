"""
J.A.R.V.I.S. Error Detector
Real-time error detection and monitoring
"""

import os

import time
import logging
from typing import Dict, List, Any


class ErrorDetector:
    """Detects errors and anomalies in real-time"""

    def __init__(self, healer):
        self.healer = healer
        self.logger = logging.getLogger('JARVIS.ErrorDetector')

    async def initialize(self):
        self.logger.info("Error Detector initialized")

    def detect_errors(self) -> List[Dict[str, Any]]:
        """Detect current errors"""
        errors = []

        try:
            # Check log files for errors
            errors.extend(self._check_log_errors())

            # Check system resources
            errors.extend(self._check_resource_errors())

            # Check running processes
            errors.extend(self._check_process_errors())

        except Exception as e:
            self.logger.error(f"Error detecting errors: {e}")

        return errors

    def _check_log_errors(self) -> List[Dict[str, Any]]:
        """Check log files for errors"""
        errors = []
        try:
            log_file = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'logs', 'jarvis.log')

            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()[-50:]  # Last 50 lines

                for line in lines:
                    if 'ERROR' in line or 'Exception' in line:
                        errors.append({
                            "type": "log_error",
                            "message": line.strip(),
                            "timestamp": time.time(),
                            "source": "log_file"
                        })

        except Exception as e:
            self.logger.error(f"Error checking log errors: {e}")

        return errors

    def _check_resource_errors(self) -> List[Dict[str, Any]]:
        """Check for resource-related errors"""
        errors = []
        try:
            if hasattr(self.healer.jarvis, 'system_monitor'):
                # Check memory
                memory_info = self.healer.jarvis.system_monitor.current_readings.get('memory', {})
                if memory_info.get('percent', 0) > 95:
                    errors.append({
                        "type": "high_memory_usage",
                        "message": f"Memory usage at {memory_info['percent']}%",
                        "timestamp": time.time(),
                        "source": "system_monitor"
                    })

                # Check CPU
                cpu_info = self.healer.jarvis.system_monitor.current_readings.get('cpu', {})
                if cpu_info.get('percent', 0) > 98:
                    errors.append({
                        "type": "high_cpu_usage",
                        "message": f"CPU usage at {cpu_info['percent']}%",
                        "timestamp": time.time(),
                        "source": "system_monitor"
                    })

        except Exception as e:
            self.logger.error(f"Error checking resource errors: {e}")

        return errors

    def _check_process_errors(self) -> List[Dict[str, Any]]:
        """Check for process-related errors"""
        errors = []
        try:
            # This would check for crashed processes, etc.
            pass
        except Exception as e:
            self.logger.error(f"Error checking process errors: {e}")

        return errors

    async def shutdown(self):
        self.logger.info("Error Detector shutdown complete")