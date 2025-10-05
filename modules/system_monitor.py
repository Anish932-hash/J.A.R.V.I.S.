"""
J.A.R.V.I.S. System Monitor
Advanced system monitoring and health tracking
"""

import os
import time
import threading
import psutil
import GPUtil
import speedtest
from typing import Dict, List, Optional, Any, Callable
import logging


class SystemMonitor:
    """
    Advanced system monitoring system
    Monitors CPU, RAM, disk, network, processes, and system health
    """

    def __init__(self, jarvis_instance):
        """
        Initialize system monitor

        Args:
            jarvis_instance: Reference to main JARVIS instance
        """
        self.jarvis = jarvis_instance
        self.logger = logging.getLogger('JARVIS.SystemMonitor')

        # Monitoring state
        self.monitoring = False
        self.monitor_thread = None

        # Monitoring intervals (seconds)
        self.intervals = {
            "cpu": 1,
            "memory": 2,
            "disk": 5,
            "network": 3,
            "processes": 10,
            "gpu": 5,
            "temperature": 30
        }

        # Historical data
        self.history = {
            "cpu": [],
            "memory": [],
            "disk": [],
            "network": [],
            "processes": [],
            "gpu": [],
            "temperature": []
        }

        # Maximum history size per metric
        self.max_history_size = 1000

        # Current readings
        self.current_readings = {}

        # Thresholds for alerts
        self.thresholds = {
            "cpu_percent": 80,
            "memory_percent": 85,
            "disk_percent": 90,
            "temperature_celsius": 70,
            "network_latency": 100
        }

        # Alert callbacks
        self.alert_callbacks = []

        # Performance tracking
        self.stats = {
            "monitoring_duration": 0,
            "alerts_triggered": 0,
            "readings_taken": 0,
            "errors_encountered": 0
        }

    def start_monitoring(self):
        """Start system monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="SystemMonitor",
                daemon=True
            )
            self.monitor_thread.start()
            self.logger.info("System monitoring started")

    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        self.logger.info("System monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        start_time = time.time()
        last_readings = {metric: 0 for metric in self.intervals.keys()}

        self.logger.info("System monitoring loop started")

        while self.monitoring:
            try:
                current_time = time.time()

                # Check each metric based on its interval
                for metric, interval in self.intervals.items():
                    if current_time - last_readings[metric] >= interval:
                        try:
                            reading = self._get_metric_reading(metric)
                            if reading:
                                self._store_reading(metric, reading)
                                self.current_readings[metric] = reading
                                self.stats["readings_taken"] += 1

                            last_readings[metric] = current_time

                        except Exception as e:
                            self.logger.error(f"Error reading {metric}: {e}")
                            self.stats["errors_encountered"] += 1

                # Check for alerts
                self._check_alerts()

                # Update monitoring duration
                self.stats["monitoring_duration"] = current_time - start_time

                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)

        self.logger.info("System monitoring loop ended")

    def _get_metric_reading(self, metric: str) -> Optional[Dict[str, Any]]:
        """Get reading for a specific metric"""
        try:
            if metric == "cpu":
                return self._get_cpu_info()
            elif metric == "memory":
                return self._get_memory_info()
            elif metric == "disk":
                return self._get_disk_info()
            elif metric == "network":
                return self._get_network_info()
            elif metric == "processes":
                return self._get_process_info()
            elif metric == "gpu":
                return self._get_gpu_info()
            elif metric == "temperature":
                return self._get_temperature_info()

        except Exception as e:
            self.logger.error(f"Error getting {metric} reading: {e}")

        return None

    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_freq = psutil.cpu_freq()
            cpu_count = psutil.cpu_count()
            cpu_count_logical = psutil.cpu_count(logical=True)

            return {
                "percent": cpu_percent,
                "frequency_mhz": cpu_freq.current if cpu_freq else 0,
                "frequency_min_mhz": cpu_freq.min if cpu_freq else 0,
                "frequency_max_mhz": cpu_freq.max if cpu_freq else 0,
                "cores_physical": cpu_count,
                "cores_logical": cpu_count_logical,
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"Error getting CPU info: {e}")
            return {}

    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            return {
                "total_bytes": memory.total,
                "available_bytes": memory.available,
                "used_bytes": memory.used,
                "free_bytes": memory.free,
                "percent": memory.percent,
                "swap_total_bytes": swap.total,
                "swap_used_bytes": swap.used,
                "swap_free_bytes": swap.free,
                "swap_percent": swap.percent,
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"Error getting memory info: {e}")
            return {}

    def _get_disk_info(self) -> Dict[str, Any]:
        """Get disk information"""
        try:
            disk = psutil.disk_usage('/')
            partitions = []

            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    partitions.append({
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "filesystem": partition.fstype,
                        "total_bytes": usage.total,
                        "used_bytes": usage.used,
                        "free_bytes": usage.free,
                        "percent": usage.percent
                    })
                except:
                    continue

            return {
                "main_total_bytes": disk.total,
                "main_used_bytes": disk.used,
                "main_free_bytes": disk.free,
                "main_percent": disk.percent,
                "partitions": partitions,
                "io_counters": self._get_disk_io_counters(),
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"Error getting disk info: {e}")
            return {}

    def _get_disk_io_counters(self) -> Dict[str, Any]:
        """Get disk I/O counters"""
        try:
            io = psutil.disk_io_counters()
            return {
                "read_bytes": io.read_bytes,
                "write_bytes": io.write_bytes,
                "read_count": io.read_count,
                "write_count": io.write_count,
                "read_time": io.read_time,
                "write_time": io.write_time
            }
        except:
            return {}

    def _get_network_info(self) -> Dict[str, Any]:
        """Get network information"""
        try:
            network = psutil.net_io_counters()
            interfaces = []

            for interface, addresses in psutil.net_if_addrs().items():
                interfaces.append({
                    "name": interface,
                    "addresses": [addr.address for addr in addresses if addr.family.name == 'AF_INET']
                })

            return {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv,
                "interfaces": interfaces,
                "connections": len(psutil.net_connections()),
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"Error getting network info: {e}")
            return {}

    def _get_process_info(self) -> Dict[str, Any]:
        """Get process information"""
        try:
            processes = []

            for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent', 'memory_percent', 'create_time']):
                try:
                    processes.append({
                        "pid": proc.info['pid'],
                        "name": proc.info['name'],
                        "status": proc.info['status'],
                        "cpu_percent": proc.info['cpu_percent'] or 0,
                        "memory_percent": proc.info['memory_percent'] or 0,
                        "create_time": proc.info['create_time']
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Sort by CPU usage
            processes.sort(key=lambda x: x['cpu_percent'], reverse=True)

            return {
                "total_processes": len(processes),
                "running_processes": len([p for p in processes if p['status'] == 'running']),
                "top_cpu_processes": processes[:10],
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"Error getting process info: {e}")
            return {}

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        try:
            gpus = GPUtil.getGPUs()
            gpu_info = []

            for gpu in gpus:
                gpu_info.append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "memory_total_mb": gpu.memoryTotal,
                    "memory_free_mb": gpu.memoryFree,
                    "memory_used_mb": gpu.memoryUsed,
                    "memory_util_percent": gpu.memoryUtil * 100,
                    "gpu_util_percent": gpu.load * 100,
                    "temperature": gpu.temperature
                })

            return {
                "gpus": gpu_info,
                "gpu_count": len(gpu_info),
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.debug(f"Error getting GPU info: {e}")
            return {"gpus": [], "gpu_count": 0}

    def _get_temperature_info(self) -> Dict[str, Any]:
        """Get temperature information"""
        try:
            # Check if sensors_temperatures is available
            if not hasattr(psutil, 'sensors_temperatures'):
                return {"sensors": {}, "note": "Temperature sensors not supported on this platform"}

            temperatures = psutil.sensors_temperatures()

            temp_info = {}
            for name, entries in temperatures.items():
                temp_info[name] = [
                    {
                        "label": entry.label,
                        "current": entry.current,
                        "high": entry.high,
                        "critical": entry.critical
                    }
                    for entry in entries
                ]

            return {
                "sensors": temp_info,
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.debug(f"Error getting temperature info: {e}")
            return {"sensors": {}}

    def _store_reading(self, metric: str, reading: Dict[str, Any]):
        """Store reading in history"""
        if metric not in self.history:
            self.history[metric] = []

        self.history[metric].append(reading)

        # Maintain history size
        if len(self.history[metric]) > self.max_history_size:
            self.history[metric].pop(0)

    def _check_alerts(self):
        """Check for threshold alerts"""
        try:
            # Skip alert checking if event manager is active
            if hasattr(self.jarvis, 'event_manager') and self.jarvis.event_manager:
                if getattr(self.jarvis.event_manager, 'is_processing', False):
                    # Event manager is active, skip local alerts
                    return
            
            # CPU alert
            cpu_info = self.current_readings.get("cpu", {})
            if cpu_info.get("percent", 0) > self.thresholds["cpu_percent"]:
                self._trigger_alert("high_cpu_usage", {
                    "metric": "cpu",
                    "value": cpu_info["percent"],
                    "threshold": self.thresholds["cpu_percent"],
                    "unit": "percent"
                })

            # Memory alert
            memory_info = self.current_readings.get("memory", {})
            if memory_info.get("percent", 0) > self.thresholds["memory_percent"]:
                self._trigger_alert("high_memory_usage", {
                    "metric": "memory",
                    "value": memory_info["percent"],
                    "threshold": self.thresholds["memory_percent"],
                    "unit": "percent"
                })

            # Disk alert
            disk_info = self.current_readings.get("disk", {})
            if disk_info.get("main_percent", 0) > self.thresholds["disk_percent"]:
                self._trigger_alert("high_disk_usage", {
                    "metric": "disk",
                    "value": disk_info["main_percent"],
                    "threshold": self.thresholds["disk_percent"],
                    "unit": "percent"
                })

            # Temperature alert
            temp_info = self.current_readings.get("temperature", {})
            for sensor_name, sensors in temp_info.get("sensors", {}).items():
                for sensor in sensors:
                    if sensor.get("current", 0) > self.thresholds["temperature_celsius"]:
                        self._trigger_alert("high_temperature", {
                            "metric": "temperature",
                            "sensor": sensor_name,
                            "label": sensor.get("label", ""),
                            "value": sensor["current"],
                            "threshold": self.thresholds["temperature_celsius"],
                            "unit": "celsius"
                        })

        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")

    def _trigger_alert(self, alert_type: str, data: Dict[str, Any]):
        """Trigger an alert"""
        try:
            # Update stats
            self.stats["alerts_triggered"] += 1

            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert_type, data)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")

            # Emit event through JARVIS event system
            event_emitted = False
            event_manager_processing = False
            
            if hasattr(self.jarvis, 'event_manager') and self.jarvis.event_manager:
                # Check if event manager is processing
                event_manager_processing = getattr(self.jarvis.event_manager, 'is_processing', False)
                
                try:
                    self.jarvis.event_manager.create_custom_event(alert_type, data)
                    event_emitted = True
                except Exception as e:
                    # Log error if event emission fails
                    self.logger.error(f"Failed to emit alert event: {alert_type} - {e}")
            
            # Completely disable warning logs for alerts
            # The event system will handle all alerts

        except Exception as e:
            self.logger.error(f"Error triggering alert: {e}")

    def get_current_readings(self) -> Dict[str, Any]:
        """Get current readings for all metrics"""
        return self.current_readings.copy()

    def get_history(self, metric: str = None, limit: int = None) -> Dict[str, Any]:
        """
        Get historical data

        Args:
            metric: Specific metric (optional)
            limit: Maximum number of readings to return

        Returns:
            Historical data
        """
        if metric:
            history = self.history.get(metric, [])
            if limit:
                history = history[-limit:]
            return {metric: history}
        else:
            result = {}
            for key, value in self.history.items():
                if limit:
                    result[key] = value[-limit:]
                else:
                    result[key] = value
            return result

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        try:
            health_score = 100
            issues = []

            # Check CPU
            cpu_info = self.current_readings.get("cpu", {})
            cpu_usage = cpu_info.get("percent", 0)
            if cpu_usage > 80:
                health_score -= 20
                issues.append(f"High CPU usage: {cpu_usage}%")
            elif cpu_usage > 60:
                health_score -= 10
                issues.append(f"Elevated CPU usage: {cpu_usage}%")

            # Check memory
            memory_info = self.current_readings.get("memory", {})
            memory_usage = memory_info.get("percent", 0)
            if memory_usage > 85:
                health_score -= 25
                issues.append(f"High memory usage: {memory_usage}%")
            elif memory_usage > 70:
                health_score -= 15
                issues.append(f"Elevated memory usage: {memory_usage}%")

            # Check disk
            disk_info = self.current_readings.get("disk", {})
            disk_usage = disk_info.get("main_percent", 0)
            if disk_usage > 90:
                health_score -= 30
                issues.append(f"Critical disk usage: {disk_usage}%")
            elif disk_usage > 80:
                health_score -= 20
                issues.append(f"High disk usage: {disk_usage}%")

            # Check temperature
            temp_info = self.current_readings.get("temperature", {})
            for sensor_name, sensors in temp_info.get("sensors", {}).items():
                for sensor in sensors:
                    temp = sensor.get("current", 0)
                    if temp > 70:
                        health_score -= 15
                        issues.append(f"High temperature ({sensor_name}): {temp}°C")

            return {
                "health_score": max(0, health_score),
                "status": "healthy" if health_score >= 80 else "warning" if health_score >= 50 else "critical",
                "issues": issues,
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"Error getting health status: {e}")
            return {"health_score": 0, "status": "error", "issues": [str(e)]}

    def run_speed_test(self) -> Dict[str, Any]:
        """Run internet speed test"""
        try:
            self.logger.info("Running internet speed test...")

            speedtester = speedtest.Speedtest()
            speedtester.get_servers()
            speedtester.get_best_server()

            download_speed = speedtester.download() / 1024 / 1024  # Convert to Mbps
            upload_speed = speedtester.upload() / 1024 / 1024      # Convert to Mbps
            ping = speedtester.results.ping

            result = {
                "download_mbps": download_speed,
                "upload_mbps": upload_speed,
                "ping_ms": ping,
                "timestamp": time.time(),
                "success": True
            }

            self.logger.info(f"Speed test completed: {download_speed:.2f} Mbps down, {upload_speed:.2f} Mbps up, {ping:.0f}ms ping")

            return result

        except Exception as e:
            self.logger.error(f"Error running speed test: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

    def get_top_processes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top processes by resource usage"""
        try:
            processes = []

            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
                try:
                    processes.append({
                        "pid": proc.pid,
                        "name": proc.name(),
                        "cpu_percent": proc.cpu_percent() or 0,
                        "memory_percent": proc.memory_percent() or 0,
                        "status": proc.status()
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Sort by CPU usage
            processes.sort(key=lambda x: x['cpu_percent'], reverse=True)

            return processes[:limit]

        except Exception as e:
            self.logger.error(f"Error getting top processes: {e}")
            return []

    def kill_process(self, pid: int) -> bool:
        """Kill a process by PID"""
        try:
            proc = psutil.Process(pid)
            proc.kill()
            self.logger.info(f"Killed process {pid}: {proc.name()}")
            return True

        except psutil.NoSuchProcess:
            self.logger.warning(f"Process {pid} not found")
            return False
        except psutil.AccessDenied:
            self.logger.error(f"Access denied killing process {pid}")
            return False
        except Exception as e:
            self.logger.error(f"Error killing process {pid}: {e}")
            return False

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        try:
            import platform
            return {
                "platform": {
                    "system": platform.system(),
                    "processor": platform.processor(),
                    "architecture": platform.architecture(),
                    "machine": platform.machine(),
                    "node": platform.node()
                },
                "boot_time": psutil.boot_time(),
                "uptime": time.time() - psutil.boot_time(),
                "current_readings": self.current_readings,
                "health_status": self.get_health_status(),
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"Error getting system info: {e}")
            return {}

    def set_threshold(self, metric: str, value: float):
        """Set alert threshold for a metric"""
        if metric in self.thresholds:
            self.thresholds[metric] = value
            self.logger.info(f"Set threshold for {metric}: {value}")

    def add_alert_callback(self, callback: Callable):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            **self.stats,
            "monitoring": self.monitoring,
            "current_readings_count": len(self.current_readings),
            "history_sizes": {metric: len(history) for metric, history in self.history.items()},
            "thresholds": self.thresholds
        }

    def clear_history(self, metric: str = None):
        """Clear monitoring history"""
        if metric:
            if metric in self.history:
                self.history[metric].clear()
                self.logger.info(f"Cleared history for {metric}")
        else:
            for key in self.history.keys():
                self.history[key].clear()
            self.logger.info("Cleared all monitoring history")

    def export_history(self, filepath: str, format: str = "json") -> bool:
        """Export monitoring history to file"""
        try:
            if format.lower() == "json":
                import json
                with open(filepath, 'w') as f:
                    json.dump(self.history, f, indent=2, default=str)
            else:
                return False

            self.logger.info(f"Exported monitoring history to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting history: {e}")
            return False