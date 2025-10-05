"""
J.A.R.V.I.S. Application Controller
Advanced application management and control system
"""

import os
import sys
import time
import subprocess
import threading
import win32gui
import win32con
import win32process
import win32api
import psutil
from typing import Dict, List, Optional, Any, Tuple
import logging


class ApplicationController:
    """
    Advanced application control system
    Can launch, close, manipulate, and monitor applications
    """

    def __init__(self, jarvis_instance):
        """
        Initialize application controller

        Args:
            jarvis_instance: Reference to main JARVIS instance
        """
        self.jarvis = jarvis_instance
        self.logger = logging.getLogger('JARVIS.ApplicationController')

        # Application tracking
        self.running_apps = {}  # pid -> app_info
        self.app_windows = {}   # window_handle -> app_info
        self.app_history = []   # Recently used applications

        # Monitoring
        self.monitoring = False
        self.monitor_thread = None

        # Application database
        self.app_database = self._load_app_database()

        # Performance tracking
        self.stats = {
            "launched_apps": 0,
            "closed_apps": 0,
            "switched_apps": 0,
            "failed_launches": 0
        }

    def initialize(self):
        """Initialize application controller"""
        try:
            self.logger.info("Initializing application controller...")

            # Start monitoring thread
            self.start_monitoring()

            # Scan currently running applications
            self._scan_running_applications()

            self.logger.info("Application controller initialized")

        except Exception as e:
            self.logger.error(f"Error initializing application controller: {e}")
            raise

    def _load_app_database(self) -> Dict[str, Any]:
        """Load application database"""
        # Default application database
        return {
            "browsers": {
                "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                "firefox": r"C:\Program Files\Mozilla Firefox\firefox.exe",
                "edge": r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
            },
            "editors": {
                "notepad": "notepad.exe",
                "vscode": r"C:\Users\{user}\AppData\Local\Programs\Microsoft VS Code\Code.exe",
                "sublime": r"C:\Program Files\Sublime Text\sublime_text.exe"
            },
            "system": {
                "calculator": "calc.exe",
                "paint": "mspaint.exe",
                "wordpad": "wordpad.exe",
                "explorer": "explorer.exe",
                "taskmgr": "taskmgr.exe",
                "control": "control.exe"
            },
            "media": {
                "vlc": r"C:\Program Files\VideoLAN\VLC\vlc.exe",
                "spotify": r"C:\Users\{user}\AppData\Roaming\Spotify\Spotify.exe",
                "wmplayer": r"C:\Program Files\Windows Media Player\wmplayer.exe"
            },
            "productivity": {
                "outlook": r"C:\Program Files\Microsoft Office\root\Office16\OUTLOOK.EXE",
                "excel": r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE",
                "word": r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
                "powerpoint": r"C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE"
            }
        }

    def _scan_running_applications(self):
        """Scan and catalog currently running applications"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'exe', 'status']):
                try:
                    app_info = {
                        "pid": proc.info['pid'],
                        "name": proc.info['name'],
                        "exe": proc.info['exe'],
                        "status": proc.info['status'],
                        "start_time": proc.create_time(),
                        "window_handle": self._find_window_for_process(proc.info['pid'])
                    }

                    self.running_apps[proc.info['pid']] = app_info

                    if app_info["window_handle"]:
                        self.app_windows[app_info["window_handle"]] = app_info

                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

            self.logger.info(f"Scanned {len(self.running_apps)} running applications")

        except Exception as e:
            self.logger.error(f"Error scanning applications: {e}")

    def _find_window_for_process(self, pid: int) -> Optional[int]:
        """Find window handle for a process"""
        try:
            def callback(hwnd, hwnds):
                if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
                    _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
                    if found_pid == pid:
                        hwnds.append(hwnd)
                return True

            hwnds = []
            win32gui.EnumWindows(callback, hwnds)
            return hwnds[0] if hwnds else None

        except Exception as e:
            self.logger.debug(f"Error finding window for PID {pid}: {e}")
            return None

    def launch_application(self,
                          app_name: str,
                          arguments: str = "",
                          working_dir: str = None,
                          wait: bool = False) -> Dict[str, Any]:
        """
        Launch an application

        Args:
            app_name: Name or path of application to launch
            arguments: Command line arguments
            working_dir: Working directory for the application
            wait: Whether to wait for application to start

        Returns:
            Launch result with process information
        """
        try:
            # Resolve application path
            app_path = self._resolve_app_path(app_name)

            if not app_path or not os.path.exists(app_path):
                return {
                    "success": False,
                    "error": f"Application not found: {app_name}",
                    "app_path": app_path
                }

            self.logger.info(f"Launching application: {app_path}")

            # Launch application
            if wait:
                process = subprocess.Popen(
                    [app_path] + arguments.split(),
                    cwd=working_dir,
                    shell=True
                )
            else:
                process = subprocess.Popen(
                    [app_path] + arguments.split(),
                    cwd=working_dir,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

            # Wait a moment for application to start
            time.sleep(1)

            # Get process info
            try:
                proc = psutil.Process(process.pid)
                app_info = {
                    "pid": process.pid,
                    "name": proc.name(),
                    "exe": app_path,
                    "status": "running",
                    "start_time": proc.create_time(),
                    "window_handle": None
                }

                # Try to find window handle
                for _ in range(10):  # Try for up to 5 seconds
                    window_handle = self._find_window_for_process(process.pid)
                    if window_handle:
                        app_info["window_handle"] = window_handle
                        break
                    time.sleep(0.5)

                # Update tracking
                self.running_apps[process.pid] = app_info
                if app_info["window_handle"]:
                    self.app_windows[app_info["window_handle"]] = app_info

                # Add to history
                self._add_to_history(app_name, "launched")

                # Update stats
                self.stats["launched_apps"] += 1

                return {
                    "success": True,
                    "message": f"Successfully launched {app_name}",
                    "process_info": app_info
                }

            except psutil.NoSuchProcess:
                return {
                    "success": False,
                    "error": "Application failed to start"
                }

        except Exception as e:
            self.logger.error(f"Error launching application {app_name}: {e}")
            self.stats["failed_launches"] += 1

            return {
                "success": False,
                "error": str(e)
            }

    def close_application(self,
                         app_identifier: str,
                         force: bool = False) -> Dict[str, Any]:
        """
        Close an application

        Args:
            app_identifier: Application name, PID, or window handle
            force: Force close if graceful close fails

        Returns:
            Close result
        """
        try:
            # Find application
            app_info = self._find_application(app_identifier)

            if not app_info:
                return {
                    "success": False,
                    "error": f"Application not found: {app_identifier}"
                }

            self.logger.info(f"Closing application: {app_info['name']} (PID: {app_info['pid']})")

            # Try graceful close first
            if not force:
                try:
                    proc = psutil.Process(app_info['pid'])
                    proc.terminate()

                    # Wait for graceful termination
                    if proc.wait(timeout=5):
                        success = True
                        message = "Application closed gracefully"
                    else:
                        success = False
                        message = "Graceful close timed out"

                except psutil.TimeoutExpired:
                    success = False
                    message = "Graceful close timed out"
                except Exception as e:
                    success = False
                    message = f"Error during graceful close: {e}"
            else:
                success = False
                message = "Force close requested"

            # Force close if graceful close failed
            if not success or force:
                try:
                    proc = psutil.Process(app_info['pid'])
                    proc.kill()
                    message = "Application force closed"
                    success = True
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Error force closing application: {e}"
                    }

            # Update tracking
            if success:
                self._remove_application(app_info['pid'])
                self._add_to_history(app_info['name'], "closed")
                self.stats["closed_apps"] += 1

            return {
                "success": success,
                "message": message,
                "app_info": app_info
            }

        except Exception as e:
            self.logger.error(f"Error closing application {app_identifier}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def switch_to_application(self, app_identifier: str) -> Dict[str, Any]:
        """
        Switch to (focus) an application

        Args:
            app_identifier: Application name, PID, or window handle

        Returns:
            Switch result
        """
        try:
            # Find application
            app_info = self._find_application(app_identifier)

            if not app_info or not app_info.get('window_handle'):
                return {
                    "success": False,
                    "error": f"Application window not found: {app_identifier}"
                }

            # Switch to window
            try:
                win32gui.ShowWindow(app_info['window_handle'], win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(app_info['window_handle'])

                # Add to history
                self._add_to_history(app_info['name'], "switched_to")
                self.stats["switched_apps"] += 1

                return {
                    "success": True,
                    "message": f"Switched to {app_info['name']}",
                    "app_info": app_info
                }

            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error switching to application: {e}"
                }

        except Exception as e:
            self.logger.error(f"Error switching to application {app_identifier}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_application_info(self, app_identifier: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an application

        Args:
            app_identifier: Application name, PID, or window handle

        Returns:
            Application information or None if not found
        """
        return self._find_application(app_identifier)

    def list_running_applications(self) -> List[Dict[str, Any]]:
        """
        List all running applications

        Returns:
            List of running application information
        """
        # Refresh application list
        self._scan_running_applications()

        return list(self.running_apps.values())

    def list_applications_by_category(self, category: str) -> List[str]:
        """
        List applications by category

        Args:
            category: Application category

        Returns:
            List of application names in category
        """
        if category in self.app_database:
            return list(self.app_database[category].keys())
        return []

    def _resolve_app_path(self, app_name: str) -> Optional[str]:
        """Resolve application name to full path"""
        # Check if it's already a full path
        if os.path.exists(app_name):
            return app_name

        # Search in database
        for category in self.app_database.values():
            if app_name.lower() in category:
                path = category[app_name.lower()]
                # Expand user variables
                path = os.path.expanduser(path.replace("{user}", os.environ.get('USERNAME', '')))
                if os.path.exists(path):
                    return path

        # Search in PATH
        try:
            return subprocess.check_output(f"where {app_name}", shell=True).decode().strip().split('\n')[0]
        except:
            pass

        return None

    def _find_application(self, app_identifier: str) -> Optional[Dict[str, Any]]:
        """Find application by name, PID, or window handle"""
        # Try PID first
        try:
            pid = int(app_identifier)
            if pid in self.running_apps:
                return self.running_apps[pid]
        except ValueError:
            pass

        # Try window handle
        try:
            hwnd = int(app_identifier)
            if hwnd in self.app_windows:
                return self.app_windows[hwnd]
        except ValueError:
            pass

        # Search by name
        app_identifier_lower = app_identifier.lower()
        for app_info in self.running_apps.values():
            if (app_info['name'].lower() == app_identifier_lower or
                app_info['exe'].lower().endswith(f"\\{app_identifier_lower}.exe")):
                return app_info

        return None

    def _remove_application(self, pid: int):
        """Remove application from tracking"""
        if pid in self.running_apps:
            app_info = self.running_apps[pid]

            # Remove from windows tracking
            if app_info.get('window_handle') and app_info['window_handle'] in self.app_windows:
                del self.app_windows[app_info['window_handle']]

            # Remove from apps tracking
            del self.running_apps[pid]

    def _add_to_history(self, app_name: str, action: str):
        """Add application action to history"""
        self.app_history.append({
            "app_name": app_name,
            "action": action,
            "timestamp": time.time()
        })

        # Keep only recent history
        if len(self.app_history) > 100:
            self.app_history.pop(0)

    def start_monitoring(self):
        """Start application monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="AppMonitor",
                daemon=True
            )
            self.monitor_thread.start()
            self.logger.info("Application monitoring started")

    def stop_monitoring(self):
        """Stop application monitoring"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
        self.logger.info("Application monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Check for crashed applications
                self._check_crashed_applications()

                # Update window handles
                self._update_window_handles()

                # Clean up dead processes
                self._cleanup_dead_processes()

                time.sleep(5)  # Monitor every 5 seconds

            except Exception as e:
                self.logger.error(f"Error in application monitoring: {e}")
                time.sleep(10)

    def _check_crashed_applications(self):
        """Check for crashed applications"""
        crashed = []

        # First pass: identify crashed applications
        for pid, app_info in list(self.running_apps.items()):
            try:
                proc = psutil.Process(pid)
                if proc.status() == psutil.STATUS_ZOMBIE:
                    crashed.append(pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                crashed.append(pid)

        # Second pass: remove crashed applications
        for pid in crashed:
            if pid in self.running_apps:
                app_info = self.running_apps[pid]
                self.logger.warning(f"Application crashed: {app_info['name']} (PID: {pid})")
                self._remove_application(pid)

    def _update_window_handles(self):
        """Update window handles for running applications"""
        for pid, app_info in list(self.running_apps.items()):
            if not app_info.get('window_handle'):
                window_handle = self._find_window_for_process(pid)
                if window_handle:
                    app_info['window_handle'] = window_handle
                    self.app_windows[window_handle] = app_info

    def _cleanup_dead_processes(self):
        """Clean up tracking for dead processes"""
        dead_processes = []

        for pid in list(self.running_apps.keys()):
            if not psutil.pid_exists(pid):
                dead_processes.append(pid)

        for pid in dead_processes:
            self._remove_application(pid)

    def get_window_info(self, window_handle: int = None) -> List[Dict[str, Any]]:
        """
        Get information about windows

        Args:
            window_handle: Specific window handle (optional)

        Returns:
            List of window information
        """
        windows_info = []

        try:
            def callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    try:
                        title = win32gui.GetWindowText(hwnd)
                        rect = win32gui.GetWindowRect(hwnd)
                        pid = win32process.GetWindowThreadProcessId(hwnd)[1]

                        window_info = {
                            "handle": hwnd,
                            "title": title,
                            "pid": pid,
                            "position": {
                                "left": rect[0],
                                "top": rect[1],
                                "right": rect[2],
                                "bottom": rect[3]
                            },
                            "size": {
                                "width": rect[2] - rect[0],
                                "height": rect[3] - rect[1]
                            }
                        }
                        windows.append(window_info)

                    except:
                        pass

                return True

            if window_handle:
                # Get specific window
                if win32gui.IsWindow(window_handle):
                    rect = win32gui.GetWindowRect(window_handle)
                    title = win32gui.GetWindowText(window_handle)

                    windows_info.append({
                        "handle": window_handle,
                        "title": title,
                        "position": {
                            "left": rect[0],
                            "top": rect[1],
                            "right": rect[2],
                            "bottom": rect[3]
                        },
                        "size": {
                            "width": rect[2] - rect[0],
                            "height": rect[3] - rect[1]
                        }
                    })
            else:
                # Get all windows
                win32gui.EnumWindows(callback, windows_info)

        except Exception as e:
            self.logger.error(f"Error getting window info: {e}")

        return windows_info

    def minimize_window(self, window_handle: int) -> bool:
        """Minimize a window"""
        try:
            win32gui.ShowWindow(window_handle, win32con.SW_MINIMIZE)
            return True
        except Exception as e:
            self.logger.error(f"Error minimizing window: {e}")
            return False

    def maximize_window(self, window_handle: int) -> bool:
        """Maximize a window"""
        try:
            win32gui.ShowWindow(window_handle, win32con.SW_MAXIMIZE)
            return True
        except Exception as e:
            self.logger.error(f"Error maximizing window: {e}")
            return False

    def resize_window(self, window_handle: int, width: int, height: int, x: int = None, y: int = None) -> bool:
        """Resize and reposition a window"""
        try:
            if x is None or y is None:
                # Get current position
                rect = win32gui.GetWindowRect(window_handle)
                x = x or rect[0]
                y = y or rect[1]

            win32gui.MoveWindow(window_handle, x, y, width, height, True)
            return True

        except Exception as e:
            self.logger.error(f"Error resizing window: {e}")
            return False

    def get_application_stats(self) -> Dict[str, Any]:
        """Get application controller statistics"""
        return {
            **self.stats,
            "running_applications": len(self.running_apps),
            "tracked_windows": len(self.app_windows),
            "history_size": len(self.app_history),
            "monitoring": self.monitoring
        }

    def get_recent_applications(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recently used applications"""
        return self.app_history[-limit:] if self.app_history else []

    def search_applications(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for applications

        Args:
            query: Search query

        Returns:
            List of matching applications
        """
        query_lower = query.lower()
        matches = []

        # Search in running applications
        for app_info in self.running_apps.values():
            if (query_lower in app_info['name'].lower() or
                query_lower in app_info['exe'].lower()):
                matches.append(app_info)

        # Search in database
        for category, apps in self.app_database.items():
            for app_name, app_path in apps.items():
                if query_lower in app_name.lower():
                    matches.append({
                        "name": app_name,
                        "path": app_path,
                        "category": category,
                        "status": "available"
                    })

        return matches

    def kill_process_tree(self, pid: int) -> bool:
        """
        Kill process and all its children

        Args:
            pid: Parent process ID

        Returns:
            Success status
        """
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)

            # Kill children first
            for child in children:
                try:
                    child.kill()
                except:
                    pass

            # Kill parent
            parent.kill()

            # Remove from tracking
            self._remove_application(pid)

            return True

        except Exception as e:
            self.logger.error(f"Error killing process tree: {e}")
            return False

    def get_process_tree(self, pid: int) -> List[Dict[str, Any]]:
        """
        Get process tree for a PID

        Args:
            pid: Process ID

        Returns:
            Process tree information
        """
        try:
            process_tree = []
            process = psutil.Process(pid)

            def add_process(proc, tree, depth=0):
                try:
                    proc_info = {
                        "pid": proc.pid,
                        "name": proc.name(),
                        "status": proc.status(),
                        "depth": depth,
                        "children": []
                    }

                    children = proc.children()
                    for child in children:
                        add_process(child, proc_info["children"], depth + 1)

                    tree.append(proc_info)

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            add_process(process, process_tree)
            return process_tree

        except Exception as e:
            self.logger.error(f"Error getting process tree: {e}")
            return []