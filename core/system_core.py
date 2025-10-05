"""
J.A.R.V.I.S. System Core
Central system management and coordination
"""

import os
import sys
import time
import json
import threading
import psutil
import platform
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

# Check for GPUtil availability
try:
    import GPUtil
    GPUUTIL_AVAILABLE = True
except ImportError:
    GPUtil = None
    GPUUTIL_AVAILABLE = False


class SystemCore:
    """
    Core system management class
    Handles system resources, performance, and coordination
    """

    def __init__(self, jarvis_instance):
        """
        Initialize system core

        Args:
            jarvis_instance: Reference to main JARVIS instance
        """
        self.jarvis = jarvis_instance
        self.system_info = self._get_system_info()
        self.resource_monitor = ResourceMonitor(self)
        self.performance_tracker = PerformanceTracker(self)
        self.hardware_detector = HardwareDetector(self)

        # System state
        self.is_initialized = False
        self.startup_time = time.time()
        self.core_threads = []

        # Callbacks
        self.health_callbacks = []
        self.resource_callbacks = []

    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        try:
            info = {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "machine": platform.machine(),
                "node": platform.node(),
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "cpu_count": os.cpu_count(),
                "ram_total": self._get_total_memory(),
                "boot_time": self._get_boot_time(),
                "current_time": datetime.now().isoformat()
            }

            # Add GPU information if available
            if GPUUTIL_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        info["gpus"] = [
                            {
                                "name": gpu.name,
                                "memory_total": gpu.memoryTotal,
                                "memory_free": gpu.memoryFree,
                                "temperature": gpu.temperature,
                                "uuid": gpu.uuid
                            } for gpu in gpus
                        ]
                    else:
                        info["gpus"] = []
                except:
                    info["gpus"] = []
            else:
                info["gpus"] = []

            return info

        except Exception as e:
            self.jarvis.logger.error(f"Error getting system info: {e}")
            return {}

    def _get_total_memory(self) -> int:
        """Get total system memory in bytes"""
        try:
            return psutil.virtual_memory().total
        except:
            return 0

    def _get_boot_time(self) -> datetime:
        """Get system boot time"""
        try:
            return datetime.fromtimestamp(psutil.boot_time())
        except:
            return datetime.now()

    def initialize(self):
        """Initialize system core"""
        try:
            self.jarvis.logger.info("Initializing system core...")

            # Start resource monitoring
            self.resource_monitor.start()

            # Start performance tracking
            self.performance_tracker.start()

            # Detect hardware capabilities
            self.hardware_detector.detect_capabilities()

            self.is_initialized = True
            self.jarvis.logger.info("System core initialized successfully")

        except Exception as e:
            self.jarvis.logger.error(f"Error initializing system core: {e}")
            raise

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "is_initialized": self.is_initialized,
            "uptime": time.time() - self.startup_time,
            "system_info": self.system_info,
            "resource_usage": self.resource_monitor.get_current_usage(),
            "performance_metrics": self.performance_tracker.get_metrics(),
            "hardware_capabilities": self.hardware_detector.get_capabilities()
        }

    def add_health_callback(self, callback: Callable):
        """Add health monitoring callback"""
        self.health_callbacks.append(callback)

    def add_resource_callback(self, callback: Callable):
        """Add resource monitoring callback"""
        self.resource_callbacks.append(callback)

    def execute_with_priority(self, func: Callable, priority: str = "normal") -> Any:
        """
        Execute function with specified priority

        Args:
            func: Function to execute
            priority: Priority level (low, normal, high, critical)
        """
        if priority == "critical":
            return self._execute_critical(func)
        elif priority == "high":
            return self._execute_high_priority(func)
        else:
            return func()

    def _execute_critical(self, func: Callable) -> Any:
        """Execute function with critical priority"""
        # Set high process priority
        try:
            current_process = psutil.Process()
            original_priority = current_process.nice()
            current_process.nice(psutil.HIGH_PRIORITY_CLASS)

            result = func()

            # Restore original priority
            current_process.nice(original_priority)
            return result

        except Exception as e:
            self.jarvis.logger.error(f"Error in critical execution: {e}")
            return func()

    def _execute_high_priority(self, func: Callable) -> Any:
        """Execute function with high priority"""
        try:
            current_process = psutil.Process()
            original_priority = current_process.nice()
            current_process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)

            result = func()

            current_process.nice(original_priority)
            return result

        except Exception as e:
            self.jarvis.logger.error(f"Error in high priority execution: {e}")
            return func()

    def optimize_performance(self):
        """Optimize system performance"""
        try:
            self.jarvis.logger.info("Optimizing system performance...")

            # Clear memory caches if possible
            self._clear_memory_cache()

            # Optimize thread priorities
            self._optimize_thread_priorities()

            # Defragment memory if needed
            self._optimize_memory_layout()

        except Exception as e:
            self.jarvis.logger.error(f"Error optimizing performance: {e}")

    def _clear_memory_cache(self):
        """Clear system memory cache"""
        try:
            # This is Windows-specific cache clearing
            import ctypes
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, 0, 0)
        except:
            pass

    def _optimize_thread_priorities(self):
        """Optimize thread priorities for better performance"""
        for thread in threading.enumerate():
            if "jarvis" in thread.name.lower():
                try:
                    thread.daemon = True
                except:
                    pass

    def _optimize_memory_layout(self):
        """Optimize memory layout for better performance"""
        try:
            import gc
            gc.collect()
        except:
            pass

    def shutdown(self):
        """Shutdown system core"""
        try:
            self.jarvis.logger.info("Shutting down system core...")

            # Stop monitoring
            if hasattr(self.resource_monitor, 'stop'):
                self.resource_monitor.stop()

            if hasattr(self.performance_tracker, 'stop'):
                self.performance_tracker.stop()

            # Wait for threads
            for thread in self.core_threads:
                if thread.is_alive():
                    thread.join(timeout=2)

            self.jarvis.logger.info("System core shutdown complete")

        except Exception as e:
            self.jarvis.logger.error(f"Error shutting down system core: {e}")


class ResourceMonitor:
    """Monitor system resources"""

    def __init__(self, system_core):
        self.system_core = system_core
        self.monitoring = False
        self.monitor_thread = None
        self.current_usage = {}

    def start(self):
        """Start resource monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="ResourceMonitor",
                daemon=True
            )
            self.monitor_thread.start()

    def stop(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)

    def get_current_usage(self) -> Dict[str, Any]:
        """Get current resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = self._get_network_usage()

            self.current_usage = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used": memory.used,
                "memory_available": memory.available,
                "disk_percent": disk.percent,
                "disk_used": disk.used,
                "disk_free": disk.free,
                "network": network,
                "timestamp": time.time()
            }

            return self.current_usage

        except Exception as e:
            self.system_core.jarvis.logger.error(f"Error getting resource usage: {e}")
            return {}

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                usage = self.get_current_usage()

                # Trigger callbacks if thresholds exceeded
                if usage.get("cpu_percent", 0) > 80:
                    self._trigger_resource_callback("high_cpu", usage)
                if usage.get("memory_percent", 0) > 85:
                    self._trigger_resource_callback("high_memory", usage)

                time.sleep(5)  # Monitor every 5 seconds

            except Exception as e:
                self.system_core.jarvis.logger.error(f"Error in monitor loop: {e}")
                time.sleep(10)

    def _get_network_usage(self) -> Dict[str, Any]:
        """Get network usage statistics"""
        try:
            network = psutil.net_io_counters()
            return {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
        except:
            return {}

    def _trigger_resource_callback(self, event_type: str, usage: Dict[str, Any]):
        """Trigger resource callbacks"""
        for callback in self.system_core.resource_callbacks:
            try:
                callback(event_type, usage)
            except Exception as e:
                self.system_core.jarvis.logger.error(f"Error in resource callback: {e}")


class PerformanceTracker:
    """Track system performance metrics"""

    def __init__(self, system_core):
        self.system_core = system_core
        self.tracking = False
        self.track_thread = None
        self.metrics_history = []
        self.max_history = 1000

    def start(self):
        """Start performance tracking"""
        if not self.tracking:
            self.tracking = True
            self.track_thread = threading.Thread(
                target=self._track_loop,
                name="PerformanceTracker",
                daemon=True
            )
            self.track_thread.start()

    def stop(self):
        """Stop performance tracking"""
        self.tracking = False
        if self.track_thread and self.track_thread.is_alive():
            self.track_thread.join(timeout=2)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            "history_length": len(self.metrics_history),
            "average_cpu": self._get_average_cpu(),
            "average_memory": self._get_average_memory(),
            "peak_cpu": self._get_peak_cpu(),
            "peak_memory": self._get_peak_memory(),
            "current_metrics": self.metrics_history[-1] if self.metrics_history else {}
        }

    def _track_loop(self):
        """Main tracking loop"""
        while self.tracking:
            try:
                metrics = {
                    "timestamp": time.time(),
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage('/').percent,
                    "thread_count": threading.active_count(),
                    "process_count": len(psutil.pids())
                }

                self.metrics_history.append(metrics)

                # Maintain history size
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history.pop(0)

                time.sleep(10)  # Track every 10 seconds

            except Exception as e:
                self.system_core.jarvis.logger.error(f"Error in track loop: {e}")
                time.sleep(30)

    def _get_average_cpu(self) -> float:
        """Get average CPU usage from history"""
        if not self.metrics_history:
            return 0.0
        return sum(m.get("cpu_percent", 0) for m in self.metrics_history) / len(self.metrics_history)

    def _get_average_memory(self) -> float:
        """Get average memory usage from history"""
        if not self.metrics_history:
            return 0.0
        return sum(m.get("memory_percent", 0) for m in self.metrics_history) / len(self.metrics_history)

    def _get_peak_cpu(self) -> float:
        """Get peak CPU usage from history"""
        if not self.metrics_history:
            return 0.0
        return max(m.get("cpu_percent", 0) for m in self.metrics_history)

    def _get_peak_memory(self) -> float:
        """Get peak memory usage from history"""
        if not self.metrics_history:
            return 0.0
        return max(m.get("memory_percent", 0) for m in self.metrics_history)


class HardwareDetector:
    """Detect hardware capabilities and features"""

    def __init__(self, system_core):
        self.system_core = system_core
        self.capabilities = {}
        self.detected_features = []

    def detect_capabilities(self):
        """Detect system hardware capabilities"""
        try:
            self.capabilities = {
                "cpu_cores": os.cpu_count() or 1,
                "has_gpu": self._detect_gpu(),
                "total_memory": psutil.virtual_memory().total,
                "has_camera": self._detect_camera(),
                "has_microphone": self._detect_microphone(),
                "has_speakers": self._detect_speakers(),
                "screen_info": self._get_screen_info(),
                "storage_devices": self._get_storage_devices(),
                "network_interfaces": self._get_network_interfaces()
            }

            self.detected_features = list(self.capabilities.keys())

        except Exception as e:
            self.system_core.jarvis.logger.error(f"Error detecting capabilities: {e}")

    def get_capabilities(self) -> Dict[str, Any]:
        """Get detected capabilities"""
        return self.capabilities

    def _detect_gpu(self) -> bool:
        """Detect if GPU is available"""
        if GPUUTIL_AVAILABLE:
            try:
                return len(GPUtil.getGPUs()) > 0
            except:
                return False
        return False

    def _detect_camera(self) -> bool:
        """Detect if camera is available"""
        try:
            import cv2
            cameras = []
            for i in range(5):  # Check first 5 camera indices
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    cameras.append(i)
                    cap.release()
            return len(cameras) > 0
        except:
            return False

    def _detect_microphone(self) -> bool:
        """Detect if microphone is available"""
        try:
            import speech_recognition as sr
            mic = sr.Microphone()
            return mic is not None
        except:
            return False

    def _detect_speakers(self) -> bool:
        """Detect if speakers are available"""
        try:
            import winsound
            return True
        except:
            return False

    def _get_screen_info(self) -> Dict[str, Any]:
        """Get screen information"""
        try:
            from screeninfo import get_monitors
            monitors = get_monitors()
            return {
                "count": len(monitors),
                "primary": {"width": monitors[0].width, "height": monitors[0].height} if monitors else {},
                "all": [{"width": m.width, "height": m.height} for m in monitors]
            }
        except:
            return {"count": 1, "primary": {"width": 1920, "height": 1080}}

    def _get_storage_devices(self) -> List[Dict[str, Any]]:
        """Get storage device information"""
        try:
            devices = []
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    devices.append({
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "filesystem": partition.fstype,
                        "total": usage.total,
                        "used": usage.used,
                        "free": usage.free
                    })
                except:
                    pass
            return devices
        except:
            return []

    def _get_network_interfaces(self) -> List[Dict[str, Any]]:
        """Get network interface information"""
        try:
            interfaces = []
            for name, stats in psutil.net_if_addrs().items():
                interfaces.append({
                    "name": name,
                    "addresses": [addr.address for addr in stats]
                })
            return interfaces
        except:
            return []