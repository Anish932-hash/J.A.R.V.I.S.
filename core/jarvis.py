"""
J.A.R.V.I.S. Main System
Advanced AI Personal Assistant for Windows
"""

import os
import sys
import time
import json
import threading
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import core modules
from jarvis.core.system_core import SystemCore
from jarvis.core.event_manager import EventManager
from jarvis.core.command_processor import CommandProcessor

# Import system modules
from jarvis.modules.voice_interface import VoiceInterface
from jarvis.modules.system_monitor import SystemMonitor
from jarvis.modules.application_controller import ApplicationController
from jarvis.modules.file_manager import FileManager
from jarvis.modules.network_manager import NetworkManager
from jarvis.modules.security_manager import SecurityManager
from jarvis.modules.plugin_manager import PluginManager

# Import advanced AI components
from .advanced.self_development_engine import SelfDevelopmentEngine
from .advanced.application_healer import ApplicationHealer
from .advanced.ethics_engine import EthicsEngine
from .advanced.memory_manager import MemoryManager
from .advanced.tester import Tester


class JARVIS:
    """
    Main J.A.R.V.I.S. system class
    Ultra-advanced AI assistant for complete PC control
    """

    def __init__(self, config_path: str = None):
        """
        Initialize J.A.R.V.I.S. system

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), '..', 'config', 'jarvis.json')
        self.is_running = False
        self.start_time = None

        # Initialize core systems
        self.logger = self._setup_logging()
        self.config = self._load_config()
        self.core = SystemCore(self)
        self.event_manager = EventManager(self)
        self.event_manager.start_processing()  # Start event processing
        self.command_processor = CommandProcessor(self)

        # Initialize modules
        self.voice_interface = None
        self.system_monitor = None
        self.app_controller = None
        self.file_manager = None
        self.network_manager = None
        self.security_manager = None
        self.plugin_manager = None

        # Initialize advanced AI components
        self.self_development_engine = None
        self.application_healer = None
        self.ethics_engine = None
        self.memory_manager = None
        self.tester = None

        # System state
        self.status = "initializing"
        self.performance_metrics = {}
        self.active_threads = []

        self.logger.info("J.A.R.V.I.S. system initialized")

    def _setup_logging(self) -> logging.Logger:
        """Setup advanced logging system"""
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)

        logger = logging.getLogger('JARVIS')
        logger.setLevel(logging.DEBUG)

        # File handler with rotation
        log_file = os.path.join(log_dir, f'jarvis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Advanced formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _load_config(self) -> Dict[str, Any]:
        """Load system configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Create default configuration
            default_config = {
                "system": {
                    "name": "J.A.R.V.I.S.",
                    "version": "2.0.0",
                    "auto_start": True,
                    "enable_voice": True,
                    "enable_gui": True,
                    "log_level": "INFO"
                },
                "voice": {
                    "engine": "windows",
                    "voice": "default",
                    "rate": 200,
                    "volume": 0.8,
                    "wake_word": "jarvis"
                },
                "monitoring": {
                    "cpu_threshold": 80,
                    "memory_threshold": 85,
                    "disk_threshold": 90,
                    "enable_alerts": True
                },
                "security": {
                    "enable_face_recognition": False,
                    "enable_voice_auth": False,
                    "encryption_enabled": True,
                    "auto_lock": False
                },
                "gui": {
                    "theme": "futuristic_dark",
                    "transparency": 0.95,
                    "animations": True,
                    "holographic_effects": True
                }
            }

            # Save default config
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=4)

            return default_config

    async def initialize_modules(self):
        """Initialize all system modules"""
        try:
            self.logger.info("Initializing system modules...")

            # Initialize voice interface
            if self.config["system"]["enable_voice"]:
                self.voice_interface = VoiceInterface(self)
                self.voice_interface.initialize()

            # Initialize system monitor
            self.system_monitor = SystemMonitor(self)
            self.system_monitor.start_monitoring()

            # Initialize application controller
            self.app_controller = ApplicationController(self)
            self.app_controller.initialize()

            # Initialize file manager
            self.file_manager = FileManager(self)
            self.file_manager.initialize()

            # Initialize network manager
            self.network_manager = NetworkManager(self)
            self.network_manager.initialize()

            # Initialize security manager
            self.security_manager = SecurityManager(self)
            self.security_manager.initialize()

            # Initialize plugin manager
            self.plugin_manager = PluginManager(self)
            self.plugin_manager.load_plugins()

            # Initialize advanced AI components
            await self._initialize_advanced_components()

            self.logger.info("All modules initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing modules: {str(e)}")
            raise

    async def _initialize_advanced_components(self):
        """Initialize advanced AI components (optimized for speed)"""
        try:
            self.logger.info("Initializing advanced AI components...")

            # Initialize only essential advanced components immediately
            # Others will be initialized lazily when needed

            # Initialize self-development engine (essential)
            self.self_development_engine = SelfDevelopmentEngine(self)
            # Initialize in background to speed up startup
            asyncio.create_task(self._init_component_async(
                self.self_development_engine, "self-development engine"
            ))

            # Initialize application healer (essential)
            self.application_healer = ApplicationHealer(self)
            # Initialize in background
            asyncio.create_task(self._init_component_async(
                self.application_healer, "application healer"
            ))

            # Initialize other components lazily (when first accessed)
            self._advanced_components_initialized = False

            self.logger.info("Advanced AI components initialization started (background)")

        except Exception as e:
            self.logger.error(f"Error initializing advanced components: {str(e)}")
            # Don't raise - allow system to continue without advanced features

    async def _init_component_async(self, component, name: str):
        """Initialize a component asynchronously"""
        try:
            await component.initialize()
            self.logger.debug(f"{name} initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize {name}: {e}")

    async def _ensure_advanced_components_ready(self):
        """Ensure all advanced components are initialized when needed"""
        if not self._advanced_components_initialized:
            try:
                # Initialize remaining components if accessed
                if not hasattr(self, 'ethics_engine') or self.ethics_engine is None:
                    self.ethics_engine = EthicsEngine(self)
                    await self.ethics_engine.initialize()

                if not hasattr(self, 'memory_manager') or self.memory_manager is None:
                    self.memory_manager = MemoryManager(self)
                    await self.memory_manager.initialize()

                if not hasattr(self, 'tester') or self.tester is None:
                    self.tester = Tester(self)
                    await self.tester.initialize()

                self._advanced_components_initialized = True
                self.logger.info("All advanced components initialized on demand")

            except Exception as e:
                self.logger.error(f"Error initializing advanced components on demand: {e}")

    async def start(self):
        """Start J.A.R.V.I.S. system"""
        try:
            self.logger.info("Starting J.A.R.V.I.S. system...")
            self.start_time = time.time()
            self.is_running = True
            self.status = "starting"

            # Initialize modules
            await self.initialize_modules()

            # Start main processing loop
            self._main_loop()

        except Exception as e:
            self.logger.error(f"Error starting J.A.R.V.I.S.: {str(e)}")
            self.shutdown()
            raise

    def _main_loop(self):
        """Main system processing loop"""
        self.status = "running"
        self.logger.info("J.A.R.V.I.S. is now running")

        try:
            while self.is_running:
                # Process events
                self.event_manager.process_events()

                # Update performance metrics
                self._update_performance_metrics()

                # Check system health
                self._check_system_health()

                # Process any pending commands
                self.command_processor.process_pending_commands()

                # Sleep to prevent excessive CPU usage
                time.sleep(0.1)

        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
            self.shutdown()
        except Exception as e:
            self.logger.error(f"Error in main loop: {str(e)}")
            self.shutdown()

    def _update_performance_metrics(self):
        """Update system performance metrics"""
        current_time = time.time()

        if not hasattr(self, '_last_metric_update'):
            self._last_metric_update = current_time
            return

        # Calculate metrics
        uptime = current_time - self.start_time
        active_threads = threading.active_count()

        self.performance_metrics = {
            "uptime": uptime,
            "active_threads": active_threads,
            "memory_usage": self._get_memory_usage(),
            "cpu_usage": self._get_cpu_usage(),
            "last_update": current_time
        }

    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 0.0

    def _check_system_health(self):
        """Check system health and trigger alerts if necessary"""
        # System health checking is now handled by the system monitor module
        # to avoid duplicate alerts. The system monitor emits events through
        # the event manager when thresholds are exceeded.
        pass

    def execute_command(self, command: str, context: Dict[str, Any] = None) -> Any:
        """
        Execute a command through the system

        Args:
            command: Command to execute
            context: Additional context for command execution

        Returns:
            Command execution result
        """
        return self.command_processor.execute_command(command, context or {})

    def speak(self, text: str, priority: str = "normal"):
        """
        Speak text through voice interface

        Args:
            text: Text to speak
            priority: Priority level (low, normal, high, critical)
        """
        if self.voice_interface:
            self.voice_interface.speak(text, priority)

    def listen(self, timeout: int = 10) -> Optional[str]:
        """
        Listen for voice input

        Args:
            timeout: Maximum time to listen in seconds

        Returns:
            Recognized text or None if no input
        """
        if self.voice_interface:
            return self.voice_interface.listen(timeout)
        return None

    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "is_running": self.is_running,
            "status": self.status,
            "uptime": time.time() - self.start_time if self.start_time else 0,
            "performance_metrics": self.performance_metrics,
            "active_modules": self._get_active_modules(),
            "version": self.config["system"]["version"]
        }

    def _get_active_modules(self) -> List[str]:
        """Get list of active modules"""
        active = []
        if self.voice_interface: active.append("voice_interface")
        if self.system_monitor: active.append("system_monitor")
        if self.app_controller: active.append("application_controller")
        if self.file_manager: active.append("file_manager")
        if self.network_manager: active.append("network_manager")
        if self.security_manager: active.append("security_manager")
        if self.plugin_manager: active.append("plugin_manager")

        # Advanced components
        if self.self_development_engine: active.append("self_development_engine")
        if self.application_healer: active.append("application_healer")
        if self.ethics_engine: active.append("ethics_engine")
        if self.memory_manager: active.append("memory_manager")
        if self.tester: active.append("tester")

        return active

    async def shutdown_async(self):
        """Async shutdown J.A.R.V.I.S. system"""
        self.logger.info("Shutting down J.A.R.V.I.S. system...")
        self.is_running = False
        self.status = "shutting_down"

        # Stop all modules
        if self.system_monitor:
            self.system_monitor.stop_monitoring()

        if self.voice_interface:
            self.voice_interface.shutdown()

        if self.plugin_manager:
            self.plugin_manager.unload_plugins()

        # Shutdown advanced components (async)
        if self.self_development_engine:
            try:
                await self.self_development_engine.shutdown()
            except:
                pass  # Ignore errors during shutdown

        if self.application_healer:
            try:
                await self.application_healer.shutdown()
            except:
                pass

        if self.ethics_engine:
            try:
                await self.ethics_engine.shutdown()
            except:
                pass

        if self.memory_manager:
            try:
                await self.memory_manager.shutdown()
            except:
                pass

        if self.tester:
            try:
                await self.tester.shutdown()
            except:
                pass

        # Wait for threads to finish
        for thread in self.active_threads:
            if thread.is_alive():
                thread.join(timeout=5)

        self.logger.info("J.A.R.V.I.S. system shutdown complete")

    def shutdown(self):
        """Shutdown J.A.R.V.I.S. system (synchronous wrapper)"""
        try:
            # Try to run async shutdown
            import asyncio
            if asyncio.get_event_loop().is_running():
                # If event loop is running, we can't use asyncio.run
                # Just do synchronous shutdown
                self._sync_shutdown()
            else:
                asyncio.run(self.shutdown_async())
        except:
            # Fallback to synchronous shutdown
            self._sync_shutdown()

    def _sync_shutdown(self):
        """Synchronous shutdown fallback"""
        self.logger.info("Performing synchronous shutdown...")
        self.is_running = False
        self.status = "shutting_down"

        # Stop all modules
        if self.system_monitor:
            self.system_monitor.stop_monitoring()

        if self.voice_interface:
            self.voice_interface.shutdown()

        if self.plugin_manager:
            self.plugin_manager.unload_plugins()

        # Shutdown advanced components (without await)
        if self.self_development_engine:
            try:
                # Call shutdown without await - ignore coroutine warnings
                self.self_development_engine.shutdown()
            except:
                pass

        if self.application_healer:
            try:
                self.application_healer.shutdown()
            except:
                pass

        if self.ethics_engine:
            try:
                self.ethics_engine.shutdown()
            except:
                pass

        if self.memory_manager:
            try:
                self.memory_manager.shutdown()
            except:
                pass

        if self.tester:
            try:
                self.tester.shutdown()
            except:
                pass

        # Wait for threads to finish
        for thread in self.active_threads:
            if thread.is_alive():
                thread.join(timeout=5)

        self.logger.info("J.A.R.V.I.S. system shutdown complete")


async def main():
    """Main entry point for J.A.R.V.I.S."""
    jarvis = JARVIS()
    try:
        await jarvis.start()
    except KeyboardInterrupt:
        jarvis.shutdown()
    except Exception as e:
        print(f"Error: {e}")
        jarvis.shutdown()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())