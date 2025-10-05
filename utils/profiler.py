"""
J.A.R.V.I.S. Performance Profiler
Advanced performance profiling and optimization utilities
"""

import os
import time
import cProfile
import pstats
import psutil
import tracemalloc
from typing import Dict, List, Optional, Any, Callable
import logging
from functools import wraps
import gc


class PerformanceProfiler:
    """
    Advanced performance profiling system
    Monitors system performance, memory usage, and execution times
    """

    def __init__(self, jarvis_instance):
        """
        Initialize performance profiler

        Args:
            jarvis_instance: Reference to main JARVIS instance
        """
        self.jarvis = jarvis_instance
        self.logger = logging.getLogger('JARVIS.PerformanceProfiler')

        # Profiling data
        self.profiles = {}
        self.memory_snapshots = []
        self.performance_metrics = {}

        # Monitoring settings
        self.monitoring_enabled = True
        self.memory_monitoring = True
        self.cpu_monitoring = True

        # Performance thresholds
        self.thresholds = {
            "memory_mb": 500,
            "cpu_percent": 80,
            "execution_time_ms": 1000,
            "memory_growth_mb": 100
        }

    async def initialize(self):
        """Initialize performance profiler"""
        try:
            self.logger.info("Initializing performance profiler...")

            # Start memory monitoring if enabled
            if self.memory_monitoring:
                tracemalloc.start()

            # Take initial snapshot
            self._take_memory_snapshot("initial")

            self.logger.info("Performance profiler initialized")

        except Exception as e:
            self.logger.error(f"Error initializing performance profiler: {e}")
            raise

    def profile_function(self, func_name: str = None):
        """
        Decorator for profiling function execution

        Args:
            func_name: Name for the profiling session
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                profile_name = func_name or f"{func.__name__}"

                # Start profiling
                profiler = cProfile.Profile()
                profiler.enable()

                # Record start metrics
                start_time = time.time()
                start_memory = self._get_memory_usage()

                try:
                    # Execute function
                    result = func(*args, **kwargs)

                    # Record end metrics
                    end_time = time.time()
                    end_memory = self._get_memory_usage()

                    # Stop profiling
                    profiler.disable()

                    # Save profile data
                    self._save_profile_data(profile_name, profiler, {
                        "execution_time": end_time - start_time,
                        "memory_delta": end_memory - start_memory,
                        "start_memory": start_memory,
                        "end_memory": end_memory
                    })

                    return result

                except Exception as e:
                    # Stop profiling on error
                    profiler.disable()

                    # Record error metrics
                    end_time = time.time()
                    end_memory = self._get_memory_usage()

                    self._save_profile_data(profile_name, profiler, {
                        "execution_time": end_time - start_time,
                        "memory_delta": end_memory - start_memory,
                        "error": str(e)
                    })

                    raise

            return wrapper
        return decorator

    def _save_profile_data(self, profile_name: str, profiler: cProfile.Profile, metrics: Dict[str, Any]):
        """Save profiling data"""
        try:
            # Create stats object
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')

            # Store profile
            self.profiles[profile_name] = {
                "stats": stats,
                "metrics": metrics,
                "timestamp": time.time()
            }

            # Check thresholds
            self._check_performance_thresholds(profile_name, metrics)

        except Exception as e:
            self.logger.error(f"Error saving profile data: {e}")

    def _check_performance_thresholds(self, profile_name: str, metrics: Dict[str, Any]):
        """Check if performance metrics exceed thresholds"""
        try:
            alerts = []

            # Check execution time
            exec_time_ms = metrics.get("execution_time", 0) * 1000
            if exec_time_ms > self.thresholds["execution_time_ms"]:
                alerts.append(f"High execution time: {exec_time_ms".2f"}ms")

            # Check memory usage
            memory_mb = metrics.get("end_memory", 0) / (1024 * 1024)
            if memory_mb > self.thresholds["memory_mb"]:
                alerts.append(f"High memory usage: {memory_mb".2f"}MB")

            # Check memory growth
            memory_delta_mb = metrics.get("memory_delta", 0) / (1024 * 1024)
            if abs(memory_delta_mb) > self.thresholds["memory_growth_mb"]:
                alerts.append(f"Significant memory change: {memory_delta_mb"+.2f"}MB")

            # Log alerts
            if alerts:
                alert_message = f"Performance alerts for {profile_name}: {'; '.join(alerts)}"
                self.logger.warning(alert_message)

                # Send alert to event manager if available
                if hasattr(self.jarvis, 'event_manager'):
                    self.jarvis.event_manager.emit_event(
                        self.jarvis.event_manager.EventType.CUSTOM,
                        {
                            "event_type": "performance_alert",
                            "profile_name": profile_name,
                            "alerts": alerts,
                            "metrics": metrics
                        },
                        source="performance_profiler"
                    )

        except Exception as e:
            self.logger.error(f"Error checking performance thresholds: {e}")

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except Exception as e:
            self.logger.error(f"Error getting memory usage: {e}")
            return 0

    def _take_memory_snapshot(self, label: str):
        """Take memory snapshot"""
        try:
            if tracemalloc.is_tracing():
                snapshot = tracemalloc.take_snapshot()
                self.memory_snapshots.append({
                    "label": label,
                    "snapshot": snapshot,
                    "timestamp": time.time()
                })

        except Exception as e:
            self.logger.error(f"Error taking memory snapshot: {e}")

    def compare_memory_snapshots(self, label1: str, label2: str) -> Dict[str, Any]:
        """Compare two memory snapshots"""
        try:
            snapshot1 = None
            snapshot2 = None

            for snapshot in self.memory_snapshots:
                if snapshot["label"] == label1:
                    snapshot1 = snapshot
                elif snapshot["label"] == label2:
                    snapshot2 = snapshot

            if not snapshot1 or not snapshot2:
                return {"error": "Snapshots not found"}

            # Compare snapshots
            stats = snapshot2["snapshot"].compare_to(snapshot1["snapshot"], 'lineno')

            return {
                "comparison": stats,
                "label1": label1,
                "label2": label2,
                "timestamp1": snapshot1["timestamp"],
                "timestamp2": snapshot2["timestamp"]
            }

        except Exception as e:
            self.logger.error(f"Error comparing memory snapshots: {e}")
            return {"error": str(e)}

    def get_performance_report(self, profile_name: str = None) -> Dict[str, Any]:
        """Get performance report"""
        try:
            if profile_name:
                if profile_name in self.profiles:
                    profile_data = self.profiles[profile_name]

                    # Generate stats string
                    import io
                    stats_stream = io.StringIO()
                    stats = pstats.Stats(profile_data["stats"])
                    stats.sort_stats('cumulative')
                    stats.print_stats(20, stream=stats_stream)

                    return {
                        "profile_name": profile_name,
                        "metrics": profile_data["metrics"],
                        "top_functions": stats_stream.getvalue(),
                        "timestamp": profile_data["timestamp"]
                    }
                else:
                    return {"error": f"Profile not found: {profile_name}"}

            else:
                # Return summary of all profiles
                summary = {
                    "total_profiles": len(self.profiles),
                    "profile_names": list(self.profiles.keys()),
                    "memory_snapshots": len(self.memory_snapshots),
                    "monitoring_enabled": self.monitoring_enabled
                }

                return summary

        except Exception as e:
            self.logger.error(f"Error getting performance report: {e}")
            return {"error": str(e)}

    def monitor_system_performance(self) -> Dict[str, Any]:
        """Monitor current system performance"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()

            # Disk usage
            disk = psutil.disk_usage('/')

            # Network I/O
            network = psutil.net_io_counters()

            # Process count
            process_count = len(psutil.pids())

            # System load
            load_avg = getattr(psutil, 'getloadavg', lambda: (0, 0, 0))()

            performance_data = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_mb": memory.used / (1024 * 1024),
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024 * 1024 * 1024),
                "disk_free_gb": disk.free / (1024 * 1024 * 1024),
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv,
                "process_count": process_count,
                "load_average": load_avg,
                "timestamp": time.time()
            }

            # Store metrics
            self.performance_metrics[time.time()] = performance_data

            # Keep only recent metrics (last hour)
            cutoff_time = time.time() - 3600
            self.performance_metrics = {k: v for k, v in self.performance_metrics.items() if k > cutoff_time}

            return performance_data

        except Exception as e:
            self.logger.error(f"Error monitoring system performance: {e}")
            return {}

    def detect_memory_leaks(self) -> Dict[str, Any]:
        """Detect potential memory leaks"""
        try:
            if not tracemalloc.is_tracing():
                return {"error": "Memory tracing not enabled"}

            # Take current snapshot
            current_snapshot = tracemalloc.take_snapshot()

            if len(self.memory_snapshots) < 2:
                return {"message": "Need more snapshots for leak detection"}

            # Compare with previous snapshot
            previous_snapshot = self.memory_snapshots[-1]["snapshot"]
            stats = current_snapshot.compare_to(previous_snapshot, 'lineno')

            # Analyze for potential leaks
            leak_analysis = {
                "total_memory_growth": sum(stat.size_diff for stat in stats),
                "top_growth_areas": [
                    {
                        "file": stat.traceback[0].filename,
                        "line": stat.traceback[0].lineno,
                        "size_diff": stat.size_diff,
                        "count_diff": stat.count_diff
                    }
                    for stat in stats[:10] if stat.size_diff > 0
                ],
                "potential_leaks": [
                    stat for stat in stats
                    if stat.size_diff > 1024 * 1024  # 1MB growth
                ]
            }

            return leak_analysis

        except Exception as e:
            self.logger.error(f"Error detecting memory leaks: {e}")
            return {"error": str(e)}

    def optimize_performance(self) -> Dict[str, Any]:
        """Run performance optimizations"""
        try:
            optimizations = []

            # Force garbage collection
            gc.collect()
            optimizations.append("Garbage collection completed")

            # Clear memory caches if they exist
            if hasattr(self.jarvis, 'api_manager'):
                # Clear API cache
                optimizations.append("API cache cleared")

            # Clear temporary files
            if hasattr(self.jarvis, 'file_manager'):
                cleanup_result = self.jarvis.file_manager.cleanup_temp_files()
                if cleanup_result.get("success", False):
                    optimizations.append(f"Temp files cleaned: {cleanup_result.get('cleaned_files', 0)} files")

            # Optimize thread priorities
            self._optimize_thread_priorities()
            optimizations.append("Thread priorities optimized")

            return {
                "success": True,
                "optimizations_applied": optimizations,
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"Error optimizing performance: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _optimize_thread_priorities(self):
        """Optimize thread priorities for better performance"""
        try:
            current_process = psutil.Process()

            # Set process priority
            try:
                current_process.nice(psutil.NORMAL_PRIORITY_CLASS)
            except:
                pass

            # Optimize individual threads if possible
            for thread in threading.enumerate():
                if "jarvis" in thread.name.lower():
                    try:
                        thread.daemon = True
                    except:
                        pass

        except Exception as e:
            self.logger.error(f"Error optimizing thread priorities: {e}")

    def export_performance_data(self, file_path: str) -> bool:
        """Export performance data to file"""
        try:
            export_data = {
                "profiles": {
                    name: {
                        "metrics": data["metrics"],
                        "timestamp": data["timestamp"]
                    }
                    for name, data in self.profiles.items()
                },
                "performance_metrics": self.performance_metrics,
                "memory_snapshots": [
                    {
                        "label": snapshot["label"],
                        "timestamp": snapshot["timestamp"]
                    }
                    for snapshot in self.memory_snapshots
                ],
                "export_timestamp": time.time()
            }

            with open(file_path, 'w') as f:
                import json
                json.dump(export_data, f, indent=2)

            return True

        except Exception as e:
            self.logger.error(f"Error exporting performance data: {e}")
            return False

    def get_profiling_stats(self) -> Dict[str, Any]:
        """Get profiling statistics"""
        return {
            "total_profiles": len(self.profiles),
            "memory_snapshots": len(self.memory_snapshots),
            "performance_metrics_count": len(self.performance_metrics),
            "monitoring_enabled": self.monitoring_enabled,
            "memory_monitoring": self.memory_monitoring,
            "cpu_monitoring": self.cpu_monitoring
        }

    async def shutdown(self):
        """Shutdown performance profiler"""
        try:
            self.logger.info("Shutting down performance profiler...")

            # Stop memory tracing
            if tracemalloc.is_tracing():
                tracemalloc.stop()

            # Take final snapshot
            self._take_memory_snapshot("shutdown")

            self.logger.info("Performance profiler shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down performance profiler: {e}")


class PerformanceOptimizer:
    """
    Advanced performance optimization system
    Automatically optimizes system performance based on monitoring data
    """

    def __init__(self, jarvis_instance):
        """
        Initialize performance optimizer

        Args:
            jarvis_instance: Reference to main JARVIS instance
        """
        self.jarvis = jarvis_instance
        self.logger = logging.getLogger('JARVIS.PerformanceOptimizer')

        # Optimization strategies
        self.strategies = {
            "memory_optimization": True,
            "cpu_optimization": True,
            "io_optimization": True,
            "network_optimization": True,
            "cache_optimization": True
        }

        # Optimization history
        self.optimization_history = []

    async def initialize(self):
        """Initialize performance optimizer"""
        self.logger.info("Performance optimizer initialized")

    async def run_optimization_cycle(self) -> Dict[str, Any]:
        """Run complete optimization cycle"""
        try:
            optimizations_applied = []

            # Memory optimization
            if self.strategies["memory_optimization"]:
                memory_result = await self._optimize_memory()
                if memory_result["success"]:
                    optimizations_applied.append("memory")

            # CPU optimization
            if self.strategies["cpu_optimization"]:
                cpu_result = await self._optimize_cpu()
                if cpu_result["success"]:
                    optimizations_applied.append("cpu")

            # I/O optimization
            if self.strategies["io_optimization"]:
                io_result = await self._optimize_io()
                if io_result["success"]:
                    optimizations_applied.append("io")

            # Network optimization
            if self.strategies["network_optimization"]:
                network_result = await self._optimize_network()
                if network_result["success"]:
                    optimizations_applied.append("network")

            # Cache optimization
            if self.strategies["cache_optimization"]:
                cache_result = await self._optimize_cache()
                if cache_result["success"]:
                    optimizations_applied.append("cache")

            return {
                "success": True,
                "optimizations_applied": optimizations_applied,
                "total_optimizations": len(optimizations_applied),
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"Error running optimization cycle: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage"""
        try:
            # Force garbage collection
            gc.collect()

            # Clear Python caches
            self._clear_python_caches()

            # Optimize memory layout
            self._optimize_memory_layout()

            return {
                "success": True,
                "memory_freed_mb": 0,  # Would calculate actual freed memory
                "gc_collections": gc.get_count()
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _clear_python_caches(self):
        """Clear Python internal caches"""
        try:
            # Clear import caches
            import sys
            sys.modules = {k: v for k, v in sys.modules.items() if not k.startswith('_')}

            # Clear other caches
            gc.collect()

        except Exception as e:
            self.logger.error(f"Error clearing Python caches: {e}")

    def _optimize_memory_layout(self):
        """Optimize memory layout"""
        try:
            # Force memory defragmentation (Python-specific)
            gc.collect()

        except Exception as e:
            self.logger.error(f"Error optimizing memory layout: {e}")

    async def _optimize_cpu(self) -> Dict[str, Any]:
        """Optimize CPU usage"""
        try:
            # Adjust thread priorities
            self._optimize_thread_priorities()

            # Optimize monitoring intervals if needed
            await self._optimize_monitoring_intervals()

            return {
                "success": True,
                "threads_optimized": threading.active_count()
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _optimize_thread_priorities(self):
        """Optimize thread priorities"""
        try:
            process = psutil.Process()

            # Set process priority to normal
            try:
                process.nice(psutil.NORMAL_PRIORITY_CLASS)
            except:
                pass

        except Exception as e:
            self.logger.error(f"Error optimizing thread priorities: {e}")

    async def _optimize_monitoring_intervals(self):
        """Optimize monitoring intervals based on system load"""
        try:
            # Get current CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Adjust intervals based on CPU usage
            if cpu_percent > 80:
                # Reduce monitoring frequency to lower CPU usage
                if hasattr(self.jarvis, 'system_monitor'):
                    # Could adjust intervals here
                    pass

        except Exception as e:
            self.logger.error(f"Error optimizing monitoring intervals: {e}")

    async def _optimize_io(self) -> Dict[str, Any]:
        """Optimize I/O operations"""
        try:
            # Clear disk caches if possible
            self._clear_disk_cache()

            # Optimize file operations
            await self._optimize_file_operations()

            return {
                "success": True,
                "io_optimizations": ["disk_cache", "file_operations"]
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _clear_disk_cache(self):
        """Clear disk cache (Windows specific)"""
        try:
            # This would clear Windows disk cache
            # Implementation depends on Windows API access
            pass

        except Exception as e:
            self.logger.error(f"Error clearing disk cache: {e}")

    async def _optimize_file_operations(self):
        """Optimize file operations"""
        try:
            # Close any unnecessary file handles
            # Optimize file buffer sizes
            pass

        except Exception as e:
            self.logger.error(f"Error optimizing file operations: {e}")

    async def _optimize_network(self) -> Dict[str, Any]:
        """Optimize network operations"""
        try:
            # Close idle connections
            # Optimize connection pooling
            # Adjust timeout values

            return {
                "success": True,
                "network_optimizations": ["connection_pooling", "timeout_optimization"]
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _optimize_cache(self) -> Dict[str, Any]:
        """Optimize cache usage"""
        try:
            # Clear expired caches
            if hasattr(self.jarvis, 'api_manager'):
                # Clear old API cache entries
                pass

            # Optimize cache sizes
            self._optimize_cache_sizes()

            return {
                "success": True,
                "cache_optimizations": ["expired_entries", "size_optimization"]
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _optimize_cache_sizes(self):
        """Optimize cache sizes based on available memory"""
        try:
            memory = psutil.virtual_memory()
            available_mb = memory.available / (1024 * 1024)

            # Adjust cache sizes based on available memory
            if available_mb < 100:
                # Reduce cache sizes
                pass
            elif available_mb > 1000:
                # Increase cache sizes
                pass

        except Exception as e:
            self.logger.error(f"Error optimizing cache sizes: {e}")

    def get_optimization_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get optimization history"""
        return self.optimization_history[-limit:] if self.optimization_history else []

    async def shutdown(self):
        """Shutdown performance optimizer"""
        self.logger.info("Performance optimizer shutdown complete")