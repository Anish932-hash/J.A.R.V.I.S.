"""
J.A.R.V.I.S. Automation Engine
Advanced Robotic Process Automation for desktop and web tasks
"""

import os
import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Tuple
import logging

# Import automation libraries
try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False

try:
    import mouse
    MOUSE_AVAILABLE = True
except ImportError:
    MOUSE_AVAILABLE = False


class AutomationTask:
    """Represents an automation task"""

    def __init__(self,
                 task_type: str,
                 description: str,
                 actions: List[Dict[str, Any]],
                 task_id: str = None):
        """
        Initialize automation task

        Args:
            task_type: Type of automation (desktop, web, keyboard, mouse)
            description: Task description
            actions: List of actions to perform
            task_id: Unique task identifier
        """
        self.task_id = task_id or f"automation_{int(time.time())}"
        self.task_type = task_type
        self.description = description
        self.actions = actions

        # Task state
        self.status = "pending"
        self.progress = 0.0
        self.current_action = 0

        # Results
        self.results = []
        self.errors = []

        # Timestamps
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "description": self.description,
            "actions": self.actions,
            "status": self.status,
            "progress": self.progress,
            "current_action": self.current_action,
            "results": self.results,
            "errors": self.errors,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }


class AutomationEngine:
    """
    Advanced automation engine for RPA
    Automates desktop applications and web browsers
    """

    def __init__(self, jarvis_instance):
        """
        Initialize automation engine

        Args:
            jarvis_instance: Reference to main JARVIS instance
        """
        self.jarvis = jarvis_instance
        self.logger = logging.getLogger('JARVIS.AutomationEngine')

        # Automation capabilities
        self.capabilities = {
            "desktop_automation": PYAUTOGUI_AVAILABLE,
            "web_automation": SELENIUM_AVAILABLE,
            "keyboard_automation": KEYBOARD_AVAILABLE,
            "mouse_automation": MOUSE_AVAILABLE
        }

        # Active automations
        self.active_tasks: Dict[str, AutomationTask] = {}
        self.completed_tasks: Dict[str, AutomationTask] = {}

        # Web drivers
        self.web_drivers = {}

        # Configuration
        self.config = {
            "screenshot_on_error": True,
            "slow_motion": False,
            "default_timeout": 10,
            "max_retries": 3,
            "screenshot_dir": os.path.join(os.path.dirname(__file__), '..', 'data', 'screenshots')
        }

        # Performance tracking
        self.stats = {
            "tasks_executed": 0,
            "actions_performed": 0,
            "automations_successful": 0,
            "automations_failed": 0,
            "total_automation_time": 0.0
        }

    async def initialize(self):
        """Initialize automation engine"""
        try:
            self.logger.info("Initializing automation engine...")

            # Create screenshot directory
            os.makedirs(self.config["screenshot_dir"], exist_ok=True)

            # Test capabilities
            await self._test_automation_capabilities()

            self.logger.info("Automation engine initialized")

        except Exception as e:
            self.logger.error(f"Error initializing automation engine: {e}")
            raise

    async def _test_automation_capabilities(self):
        """Test automation capabilities"""
        try:
            if PYAUTOGUI_AVAILABLE:
                # Test screen size
                screen_size = pyautogui.size()
                self.logger.info(f"✓ Desktop automation: {screen_size}")

            if SELENIUM_AVAILABLE:
                # Test Chrome driver
                try:
                    options = Options()
                    options.add_argument("--headless")
                    driver = webdriver.Chrome(options=options)
                    driver.quit()
                    self.logger.info("✓ Web automation available")
                except Exception as e:
                    self.logger.warning(f"✗ Web automation test failed: {e}")

        except Exception as e:
            self.logger.error(f"Error testing automation capabilities: {e}")

    async def create_task(self,
                         task_type: str,
                         description: str,
                         actions: List[Dict[str, Any]]) -> str:
        """
        Create automation task

        Args:
            task_type: Type of automation task
            description: Task description
            actions: List of actions to perform

        Returns:
            Task ID
        """
        try:
            task = AutomationTask(task_type, description, actions)

            self.active_tasks[task.task_id] = task

            self.logger.info(f"Created automation task: {task.task_id}")

            # Start execution
            asyncio.create_task(self._execute_task(task))

            return task.task_id

        except Exception as e:
            self.logger.error(f"Error creating automation task: {e}")
            return ""

    async def _execute_task(self, task: AutomationTask):
        """Execute automation task"""
        try:
            task.status = "running"
            task.started_at = time.time()

            self.logger.info(f"Executing automation task: {task.task_id}")

            total_actions = len(task.actions)

            for i, action in enumerate(task.actions):
                try:
                    task.current_action = i
                    task.progress = (i / total_actions) * 100

                    # Execute action based on type
                    if task.task_type == "desktop":
                        await self._execute_desktop_action(action)
                    elif task.task_type == "web":
                        await self._execute_web_action(action)
                    elif task.task_type == "keyboard":
                        await self._execute_keyboard_action(action)
                    elif task.task_type == "mouse":
                        await self._execute_mouse_action(action)

                    task.results.append({
                        "action": action,
                        "success": True,
                        "timestamp": time.time()
                    })

                    self.stats["actions_performed"] += 1

                    # Slow motion delay if enabled
                    if self.config["slow_motion"]:
                        await asyncio.sleep(1)

                except Exception as e:
                    error_msg = f"Error in action {i}: {str(e)}"
                    task.errors.append(error_msg)

                    if self.config["screenshot_on_error"]:
                        await self._take_screenshot(f"error_{task.task_id}_{i}")

                    self.logger.error(error_msg)

                    # Continue with next action or fail task
                    if action.get("critical", False):
                        task.status = "failed"
                        break

            # Task completed
            task.status = "completed" if not task.errors else "failed"
            task.completed_at = time.time()
            task.progress = 100.0

            # Update stats
            self.stats["tasks_executed"] += 1
            if task.status == "completed":
                self.stats["automations_successful"] += 1
            else:
                self.stats["automations_failed"] += 1

            # Move to completed
            self.completed_tasks[task.task_id] = task
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]

            self.logger.info(f"Automation task {task.task_id} {'completed' if task.status == 'completed' else 'failed'}")

        except Exception as e:
            self.logger.error(f"Error executing automation task {task.task_id}: {e}")
            task.status = "failed"
            task.errors.append(str(e))

    async def _execute_desktop_action(self, action: Dict[str, Any]):
        """Execute desktop automation action"""
        try:
            if not PYAUTOGUI_AVAILABLE:
                raise Exception("Desktop automation not available")

            action_type = action.get("type", "")

            if action_type == "click":
                x, y = action.get("position", (0, 0))
                pyautogui.click(x, y)

            elif action_type == "type":
                text = action.get("text", "")
                pyautogui.typewrite(text)

            elif action_type == "press_key":
                key = action.get("key", "")
                pyautogui.press(key)

            elif action_type == "screenshot":
                filename = action.get("filename", f"screenshot_{int(time.time())}.png")
                filepath = os.path.join(self.config["screenshot_dir"], filename)
                pyautogui.screenshot(filepath)

            elif action_type == "wait":
                seconds = action.get("seconds", 1)
                await asyncio.sleep(seconds)

            elif action_type == "move_to":
                x, y = action.get("position", (0, 0))
                pyautogui.moveTo(x, y)

        except Exception as e:
            self.logger.error(f"Error executing desktop action: {e}")
            raise

    async def _execute_web_action(self, action: Dict[str, Any]):
        """Execute web automation action"""
        try:
            if not SELENIUM_AVAILABLE:
                raise Exception("Web automation not available")

            driver_id = action.get("driver_id", "default")
            driver = self._get_web_driver(driver_id)

            action_type = action.get("type", "")

            if action_type == "navigate":
                url = action.get("url", "")
                driver.get(url)

            elif action_type == "click":
                selector = action.get("selector", "")
                by_type = action.get("by", "id")

                if by_type == "id":
                    element = driver.find_element(By.ID, selector)
                elif by_type == "class":
                    element = driver.find_element(By.CLASS_NAME, selector)
                elif by_type == "xpath":
                    element = driver.find_element(By.XPATH, selector)
                else:
                    element = driver.find_element(By.CSS_SELECTOR, selector)

                element.click()

            elif action_type == "type":
                selector = action.get("selector", "")
                text = action.get("text", "")
                by_type = action.get("by", "id")

                if by_type == "id":
                    element = driver.find_element(By.ID, selector)
                elif by_type == "class":
                    element = driver.find_element(By.CLASS_NAME, selector)
                elif by_type == "xpath":
                    element = driver.find_element(By.XPATH, selector)
                else:
                    element = driver.find_element(By.CSS_SELECTOR, selector)

                element.clear()
                element.send_keys(text)

            elif action_type == "wait_for_element":
                selector = action.get("selector", "")
                timeout = action.get("timeout", self.config["default_timeout"])
                by_type = action.get("by", "id")

                wait = WebDriverWait(driver, timeout)

                if by_type == "id":
                    wait.until(EC.presence_of_element_located((By.ID, selector)))
                elif by_type == "class":
                    wait.until(EC.presence_of_element_located((By.CLASS_NAME, selector)))
                elif by_type == "xpath":
                    wait.until(EC.presence_of_element_located((By.XPATH, selector)))
                else:
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))

            elif action_type == "screenshot":
                filename = action.get("filename", f"web_screenshot_{int(time.time())}.png")
                filepath = os.path.join(self.config["screenshot_dir"], filename)
                driver.save_screenshot(filepath)

        except Exception as e:
            self.logger.error(f"Error executing web action: {e}")
            raise

    async def _execute_keyboard_action(self, action: Dict[str, Any]):
        """Execute keyboard automation action"""
        try:
            if not KEYBOARD_AVAILABLE:
                raise Exception("Keyboard automation not available")

            action_type = action.get("type", "")

            if action_type == "press":
                key = action.get("key", "")
                keyboard.press(key)

            elif action_type == "release":
                key = action.get("key", "")
                keyboard.release(key)

            elif action_type == "type":
                text = action.get("text", "")
                keyboard.write(text)

            elif action_type == "hotkey":
                keys = action.get("keys", [])
                keyboard.press_and_release('+'.join(keys))

        except Exception as e:
            self.logger.error(f"Error executing keyboard action: {e}")
            raise

    async def _execute_mouse_action(self, action: Dict[str, Any]):
        """Execute mouse automation action"""
        try:
            if not MOUSE_AVAILABLE:
                raise Exception("Mouse automation not available")

            action_type = action.get("type", "")

            if action_type == "move":
                x, y = action.get("position", (0, 0))
                mouse.move(x, y)

            elif action_type == "click":
                button = action.get("button", "left")
                mouse.click(button)

            elif action_type == "scroll":
                x, y = action.get("direction", (0, 1))
                mouse.wheel(x, y)

        except Exception as e:
            self.logger.error(f"Error executing mouse action: {e}")
            raise

    def _get_web_driver(self, driver_id: str = "default"):
        """Get or create web driver"""
        try:
            if driver_id not in self.web_drivers:
                options = Options()
                options.add_argument("--headless")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")

                driver = webdriver.Chrome(options=options)
                self.web_drivers[driver_id] = driver

            return self.web_drivers[driver_id]

        except Exception as e:
            self.logger.error(f"Error getting web driver: {e}")
            raise

    async def _take_screenshot(self, filename: str):
        """Take screenshot"""
        try:
            if PYAUTOGUI_AVAILABLE:
                filepath = os.path.join(self.config["screenshot_dir"], filename)
                pyautogui.screenshot(filepath)
                self.logger.info(f"Screenshot saved: {filepath}")

        except Exception as e:
            self.logger.error(f"Error taking screenshot: {e}")

    async def open_application(self, app_path: str) -> Dict[str, Any]:
        """Open desktop application"""
        try:
            if not os.path.exists(app_path):
                return {
                    "success": False,
                    "error": f"Application not found: {app_path}"
                }

            # Use os.startfile for Windows
            os.startfile(app_path)

            return {
                "success": True,
                "message": f"Opened application: {app_path}"
            }

        except Exception as e:
            self.logger.error(f"Error opening application: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def automate_web_task(self,
                               url: str,
                               actions: List[Dict[str, Any]],
                               driver_id: str = "default") -> Dict[str, Any]:
        """Automate web-based task"""
        try:
            # Create automation task
            task = AutomationTask("web", f"Web automation for {url}", actions)
            self.active_tasks[task.task_id] = task

            # Execute task
            await self._execute_task(task)

            return {
                "success": task.status == "completed",
                "task_id": task.task_id,
                "results": task.results,
                "errors": task.errors
            }

        except Exception as e:
            self.logger.error(f"Error automating web task: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def automate_desktop_task(self,
                                   actions: List[Dict[str, Any]],
                                   description: str = "Desktop automation") -> Dict[str, Any]:
        """Automate desktop-based task"""
        try:
            # Create automation task
            task = AutomationTask("desktop", description, actions)
            self.active_tasks[task.task_id] = task

            # Execute task
            await self._execute_task(task)

            return {
                "success": task.status == "completed",
                "task_id": task.task_id,
                "results": task.results,
                "errors": task.errors
            }

        except Exception as e:
            self.logger.error(f"Error automating desktop task: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen size"""
        try:
            if PYAUTOGUI_AVAILABLE:
                return pyautogui.size()
            return (1920, 1080)  # Default fallback

        except Exception as e:
            self.logger.error(f"Error getting screen size: {e}")
            return (1920, 1080)

    def get_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse position"""
        try:
            if PYAUTOGUI_AVAILABLE:
                return pyautogui.position()
            return (0, 0)

        except Exception as e:
            self.logger.error(f"Error getting mouse position: {e}")
            return (0, 0)

    def take_screenshot(self, filename: str = None) -> str:
        """Take screenshot"""
        try:
            if filename is None:
                filename = f"screenshot_{int(time.time())}.png"

            filepath = os.path.join(self.config["screenshot_dir"], filename)

            if PYAUTOGUI_AVAILABLE:
                pyautogui.screenshot(filepath)

            return filepath

        except Exception as e:
            self.logger.error(f"Error taking screenshot: {e}")
            return ""

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get automation task status"""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id].to_dict()
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id].to_dict()
        else:
            return None

    def get_all_tasks(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all automation tasks"""
        return {
            "active": [task.to_dict() for task in self.active_tasks.values()],
            "completed": [task.to_dict() for task in self.completed_tasks.values()]
        }

    def cancel_task(self, task_id: str) -> bool:
        """Cancel automation task"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = "cancelled"
            task.completed_at = time.time()

            # Move to completed
            self.completed_tasks[task_id] = task
            del self.active_tasks[task_id]

            return True
        return False

    def get_capabilities(self) -> Dict[str, bool]:
        """Get automation capabilities"""
        return self.capabilities.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get automation statistics"""
        return {
            **self.stats,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "capabilities": self.capabilities
        }

    async def shutdown(self):
        """Shutdown automation engine"""
        try:
            self.logger.info("Shutting down automation engine...")

            # Close web drivers
            for driver in self.web_drivers.values():
                try:
                    driver.quit()
                except:
                    pass

            self.web_drivers.clear()

            self.logger.info("Automation engine shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down automation engine: {e}")

    # Predefined automation templates

    async def automate_email_sending(self,
                                   recipient: str,
                                   subject: str,
                                   body: str,
                                   email_app: str = "outlook") -> Dict[str, Any]:
        """Automate email sending"""
        try:
            actions = []

            if email_app.lower() == "outlook":
                # Outlook automation actions
                actions = [
                    {"type": "click", "position": (100, 100)},  # New email button (approximate)
                    {"type": "type", "text": recipient},  # To field
                    {"type": "press_key", "key": "tab"},  # Move to subject
                    {"type": "type", "text": subject},  # Subject field
                    {"type": "press_key", "key": "tab"},  # Move to body
                    {"type": "type", "text": body},  # Email body
                    {"type": "press_key", "key": "ctrl+enter"}  # Send email
                ]

            return await self.automate_desktop_task(actions, f"Send email to {recipient}")

        except Exception as e:
            self.logger.error(f"Error automating email: {e}")
            return {"success": False, "error": str(e)}

    async def automate_file_organization(self,
                                       source_folder: str,
                                       file_types: Dict[str, str]) -> Dict[str, Any]:
        """Automate file organization by moving files to appropriate folders based on type"""
        try:
            import shutil
            import mimetypes

            if not os.path.exists(source_folder):
                return {
                    "success": False,
                    "error": f"Source folder does not exist: {source_folder}"
                }

            files_moved = 0
            folders_created = 0

            # Scan all files in source folder
            for filename in os.listdir(source_folder):
                filepath = os.path.join(source_folder, filename)

                if not os.path.isfile(filepath):
                    continue

                # Determine file type
                file_ext = os.path.splitext(filename)[1].lower()
                target_folder = None

                # Check file extension mapping
                for ext, folder in file_types.items():
                    if file_ext == f'.{ext.lower()}':
                        target_folder = folder
                        break

                # If not found in extension mapping, try MIME type
                if target_folder is None:
                    mime_type, _ = mimetypes.guess_type(filename)
                    if mime_type:
                        if mime_type.startswith('image/'):
                            target_folder = file_types.get('images', 'Images')
                        elif mime_type.startswith('video/'):
                            target_folder = file_types.get('videos', 'Videos')
                        elif mime_type.startswith('audio/'):
                            target_folder = file_types.get('audio', 'Audio')
                        elif mime_type.startswith('text/'):
                            target_folder = file_types.get('documents', 'Documents')
                        elif mime_type in ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                            target_folder = file_types.get('documents', 'Documents')

                # Default to 'Other' if no match
                if target_folder is None:
                    target_folder = file_types.get('other', 'Other')

                # Create target folder if it doesn't exist
                target_path = os.path.join(source_folder, target_folder)
                if not os.path.exists(target_path):
                    os.makedirs(target_path)
                    folders_created += 1

                # Move file
                target_file = os.path.join(target_path, filename)
                shutil.move(filepath, target_file)
                files_moved += 1

                self.logger.debug(f"Moved {filename} to {target_folder}")

            return {
                "success": True,
                "message": f"File organization completed successfully",
                "files_moved": files_moved,
                "folders_created": folders_created,
                "source_folder": source_folder
            }

        except Exception as e:
            self.logger.error(f"Error automating file organization: {e}")
            return {"success": False, "error": str(e)}

    async def automate_data_entry(self,
                                application: str,
                                data_fields: Dict[str, str]) -> Dict[str, Any]:
        """Automate data entry in applications"""
        try:
            actions = []

            # Generate actions based on application type
            if application.lower() == "excel":
                # Excel data entry automation
                for field, value in data_fields.items():
                    actions.append({"type": "type", "text": value})
                    actions.append({"type": "press_key", "key": "enter"})

            return await self.automate_desktop_task(actions, f"Data entry in {application}")

        except Exception as e:
            self.logger.error(f"Error automating data entry: {e}")
            return {"success": False, "error": str(e)}