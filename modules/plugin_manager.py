"""
J.A.R.V.I.S. Plugin Manager
Advanced plugin system for extensibility
"""

import os
import sys
import time
import importlib
import inspect
from typing import Dict, List, Optional, Any, Callable
import logging


class Plugin:
    """Plugin class for J.A.R.V.I.S."""

    def __init__(self, name: str, version: str, description: str, author: str):
        """
        Initialize plugin

        Args:
            name: Plugin name
            version: Plugin version
            description: Plugin description
            author: Plugin author
        """
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.enabled = False
        self.loaded = False
        self.module = None
        self.load_time = None
        self.dependencies = []
        self.commands = []
        self.hooks = {}

    def enable(self):
        """Enable plugin"""
        self.enabled = True

    def disable(self):
        """Disable plugin"""
        self.enabled = False

    def is_enabled(self) -> bool:
        """Check if plugin is enabled"""
        return self.enabled and self.loaded


class PluginManager:
    """
    Advanced plugin management system
    Handles plugin loading, unloading, and coordination
    """

    def __init__(self, jarvis_instance):
        """
        Initialize plugin manager

        Args:
            jarvis_instance: Reference to main JARVIS instance
        """
        self.jarvis = jarvis_instance
        self.logger = logging.getLogger('JARVIS.PluginManager')

        # Plugin storage
        self.plugins = {}  # name -> Plugin instance
        self.loaded_plugins = {}  # name -> module

        # Plugin directories
        self.plugin_directories = [
            os.path.join(os.path.dirname(__file__), '..', 'plugins'),
            os.path.join(os.path.dirname(__file__), '..', 'modules')
        ]

        # Plugin hooks
        self.hooks = {
            "on_command": [],
            "on_speech": [],
            "on_system_startup": [],
            "on_system_shutdown": [],
            "on_file_operation": [],
            "on_network_event": [],
            "on_security_event": []
        }

        # Plugin statistics
        self.stats = {
            "plugins_loaded": 0,
            "plugins_failed": 0,
            "commands_registered": 0,
            "hooks_registered": 0
        }

    def load_plugins(self):
        """Load all available plugins"""
        try:
            self.logger.info("Loading plugins...")

            for plugin_dir in self.plugin_directories:
                if os.path.exists(plugin_dir):
                    self._scan_plugin_directory(plugin_dir)

            # Load core modules as plugins
            self._load_core_modules()

            self.logger.info(f"Loaded {self.stats['plugins_loaded']} plugins")

        except Exception as e:
            self.logger.error(f"Error loading plugins: {e}")

    def _scan_plugin_directory(self, directory: str):
        """Scan directory for plugins"""
        try:
            for item in os.listdir(directory):
                plugin_path = os.path.join(directory, item)

                # Check if it's a plugin file or directory
                if item.endswith('.py') and not item.startswith('__'):
                    self._load_plugin_file(plugin_path)
                elif os.path.isdir(plugin_path) and not item.startswith('__'):
                    self._load_plugin_directory(plugin_path)

        except Exception as e:
            self.logger.error(f"Error scanning plugin directory {directory}: {e}")

    def _load_plugin_file(self, plugin_path: str):
        """Load a single plugin file"""
        try:
            plugin_name = os.path.splitext(os.path.basename(plugin_path))[0]

            # Import the plugin module
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
            if not spec:
                return

            module = importlib.util.module_from_spec(spec)

            # Add to sys.modules before executing
            sys.modules[plugin_name] = module

            # Execute the module
            spec.loader.exec_module(module)

            # Check if module has plugin class
            if hasattr(module, 'JARVISPlugin'):
                plugin_class = module.JARVISPlugin
                self._register_plugin_class(plugin_name, plugin_class)

        except Exception as e:
            self.logger.error(f"Error loading plugin file {plugin_path}: {e}")
            self.stats["plugins_failed"] += 1

    def _load_plugin_directory(self, plugin_dir: str):
        """Load plugin from directory"""
        try:
            plugin_name = os.path.basename(plugin_dir)

            # Look for __init__.py
            init_file = os.path.join(plugin_dir, '__init__.py')
            if not os.path.exists(init_file):
                return

            # Import the plugin package
            package_name = f"jarvis.plugins.{plugin_name}"

            try:
                module = importlib.import_module(package_name)

                if hasattr(module, 'JARVISPlugin'):
                    plugin_class = module.JARVISPlugin
                    self._register_plugin_class(plugin_name, plugin_class)

            except ImportError as e:
                self.logger.error(f"Error importing plugin package {package_name}: {e}")
                self.stats["plugins_failed"] += 1

        except Exception as e:
            self.logger.error(f"Error loading plugin directory {plugin_dir}: {e}")
            self.stats["plugins_failed"] += 1

    def _register_plugin_class(self, plugin_name: str, plugin_class: type):
        """Register a plugin class"""
        try:
            # Create plugin instance
            plugin = plugin_class()

            # Get plugin metadata
            plugin_info = {
                "name": getattr(plugin, 'name', plugin_name),
                "version": getattr(plugin, 'version', '1.0.0'),
                "description": getattr(plugin, 'description', 'No description'),
                "author": getattr(plugin, 'author', 'Unknown')
            }

            # Create plugin object
            jarvis_plugin = Plugin(
                plugin_info["name"],
                plugin_info["version"],
                plugin_info["description"],
                plugin_info["author"]
            )

            # Store module reference
            jarvis_plugin.module = plugin_class
            jarvis_plugin.dependencies = getattr(plugin, 'dependencies', [])
            jarvis_plugin.commands = getattr(plugin, 'commands', [])
            jarvis_plugin.hooks = getattr(plugin, 'hooks', {})

            # Register plugin
            self.plugins[plugin_name] = jarvis_plugin
            self.loaded_plugins[plugin_name] = plugin_class

            # Register hooks
            self._register_plugin_hooks(jarvis_plugin)

            # Register commands
            self._register_plugin_commands(jarvis_plugin)

            self.stats["plugins_loaded"] += 1
            self.logger.info(f"Loaded plugin: {plugin_name} v{plugin_info['version']}")

        except Exception as e:
            self.logger.error(f"Error registering plugin {plugin_name}: {e}")
            self.stats["plugins_failed"] += 1

    def _register_plugin_hooks(self, plugin: Plugin):
        """Register plugin hooks"""
        try:
            for hook_name, hook_function in plugin.hooks.items():
                if hook_name in self.hooks:
                    self.hooks[hook_name].append({
                        "plugin": plugin.name,
                        "function": hook_function
                    })
                    self.stats["hooks_registered"] += 1

        except Exception as e:
            self.logger.error(f"Error registering hooks for plugin {plugin.name}: {e}")

    def _register_plugin_commands(self, plugin: Plugin):
        """Register plugin commands"""
        try:
            for command in plugin.commands:
                if "pattern" in command and "handler" in command:
                    self.jarvis.command_processor.register_command_pattern(
                        command["pattern"],
                        f"plugin.{plugin.name}.{command.get('name', 'unknown')}",
                        command.get("entities", []),
                        command["handler"]
                    )
                    self.stats["commands_registered"] += 1

        except Exception as e:
            self.logger.error(f"Error registering commands for plugin {plugin.name}: {e}")

    def _load_core_modules(self):
        """Load core modules as plugins"""
        try:
            # Register core modules as built-in plugins
            core_modules = [
                ("voice_interface", "Voice Interface", "1.0.0", "J.A.R.V.I.S. Team"),
                ("system_monitor", "System Monitor", "1.0.0", "J.A.R.V.I.S. Team"),
                ("application_controller", "Application Controller", "1.0.0", "J.A.R.V.I.S. Team"),
                ("file_manager", "File Manager", "1.0.0", "J.A.R.V.I.S. Team"),
                ("network_manager", "Network Manager", "1.0.0", "J.A.R.V.I.S. Team"),
                ("security_manager", "Security Manager", "1.0.0", "J.A.R.V.I.S. Team")
            ]

            for module_name, description, version, author in core_modules:
                plugin = Plugin(module_name, version, description, author)
                plugin.enabled = True
                plugin.loaded = True
                plugin.commands = []  # Core modules register commands directly
                plugin.hooks = {}

                self.plugins[module_name] = plugin
                self.stats["plugins_loaded"] += 1

        except Exception as e:
            self.logger.error(f"Error loading core modules: {e}")

    def enable_plugin(self, plugin_name: str) -> bool:
        """
        Enable a plugin

        Args:
            plugin_name: Name of plugin to enable

        Returns:
            Success status
        """
        try:
            if plugin_name not in self.plugins:
                return False

            plugin = self.plugins[plugin_name]

            if not plugin.loaded:
                return False

            plugin.enable()

            # Call plugin's on_enable method if it exists
            if hasattr(plugin.module, 'on_enable'):
                try:
                    plugin.module.on_enable()
                except Exception as e:
                    self.logger.error(f"Error in plugin {plugin_name} on_enable: {e}")

            self.logger.info(f"Enabled plugin: {plugin_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error enabling plugin {plugin_name}: {e}")
            return False

    def disable_plugin(self, plugin_name: str) -> bool:
        """
        Disable a plugin

        Args:
            plugin_name: Name of plugin to disable

        Returns:
            Success status
        """
        try:
            if plugin_name not in self.plugins:
                return False

            plugin = self.plugins[plugin_name]
            plugin.disable()

            # Call plugin's on_disable method if it exists
            if hasattr(plugin.module, 'on_disable'):
                try:
                    plugin.module.on_disable()
                except Exception as e:
                    self.logger.error(f"Error in plugin {plugin_name} on_disable: {e}")

            self.logger.info(f"Disabled plugin: {plugin_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error disabling plugin {plugin_name}: {e}")
            return False

    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin

        Args:
            plugin_name: Name of plugin to unload

        Returns:
            Success status
        """
        try:
            if plugin_name not in self.plugins:
                return False

            plugin = self.plugins[plugin_name]

            # Disable first
            if plugin.enabled:
                self.disable_plugin(plugin_name)

            # Remove from loaded plugins
            if plugin_name in self.loaded_plugins:
                del self.loaded_plugins[plugin_name]

            # Remove hooks
            self._remove_plugin_hooks(plugin)

            # Remove commands
            self._remove_plugin_commands(plugin)

            plugin.loaded = False

            self.logger.info(f"Unloaded plugin: {plugin_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False

    def _remove_plugin_hooks(self, plugin: Plugin):
        """Remove plugin hooks"""
        try:
            for hook_name in self.hooks.keys():
                self.hooks[hook_name] = [
                    hook for hook in self.hooks[hook_name]
                    if hook["plugin"] != plugin.name
                ]

        except Exception as e:
            self.logger.error(f"Error removing hooks for plugin {plugin.name}: {e}")

    def _remove_plugin_commands(self, plugin: Plugin):
        """Remove plugin commands"""
        try:
            # Remove commands from command processor
            for command in plugin.commands:
                if "pattern" in command:
                    # Unregister the command pattern
                    command_name = f"plugin.{plugin.name}.{command.get('name', 'unknown')}"
                    # This assumes command_processor has an unregister method
                    if hasattr(self.jarvis.command_processor, 'unregister_command_pattern'):
                        self.jarvis.command_processor.unregister_command_pattern(command["pattern"], command_name)

            self.logger.debug(f"Removed {len(plugin.commands)} commands for plugin {plugin.name}")

        except Exception as e:
            self.logger.error(f"Error removing commands for plugin {plugin.name}: {e}")

    def call_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """
        Call a plugin hook

        Args:
            hook_name: Name of hook to call
            *args: Arguments to pass to hook functions
            **kwargs: Keyword arguments to pass to hook functions

        Returns:
            List of hook return values
        """
        results = []

        if hook_name in self.hooks:
            for hook_info in self.hooks[hook_name]:
                try:
                    if hook_info["function"]:
                        result = hook_info["function"](*args, **kwargs)
                        results.append(result)

                except Exception as e:
                    self.logger.error(f"Error in hook {hook_name} for plugin {hook_info['plugin']}: {e}")

        return results

    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a plugin

        Args:
            plugin_name: Name of plugin

        Returns:
            Plugin information or None if not found
        """
        if plugin_name not in self.plugins:
            return None

        plugin = self.plugins[plugin_name]

        return {
            "name": plugin.name,
            "version": plugin.version,
            "description": plugin.description,
            "author": plugin.author,
            "enabled": plugin.enabled,
            "loaded": plugin.loaded,
            "load_time": plugin.load_time,
            "dependencies": plugin.dependencies,
            "commands": plugin.commands,
            "hooks": list(plugin.hooks.keys())
        }

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all plugins"""
        return [self.get_plugin_info(name) for name in self.plugins.keys()]

    def list_enabled_plugins(self) -> List[Dict[str, Any]]:
        """List enabled plugins"""
        return [self.get_plugin_info(name) for name, plugin in self.plugins.items() if plugin.enabled]

    def list_disabled_plugins(self) -> List[Dict[str, Any]]:
        """List disabled plugins"""
        return [self.get_plugin_info(name) for name, plugin in self.plugins.items() if not plugin.enabled]

    def reload_plugin(self, plugin_name: str) -> bool:
        """
        Reload a plugin

        Args:
            plugin_name: Name of plugin to reload

        Returns:
            Success status
        """
        try:
            # Unload first
            if not self.unload_plugin(plugin_name):
                return False

            # Reload by loading again
            plugin_dir = None
            for directory in self.plugin_directories:
                plugin_path = os.path.join(directory, plugin_name + '.py')
                if os.path.exists(plugin_path):
                    plugin_dir = directory
                    break

            if plugin_dir:
                self._scan_plugin_directory(plugin_dir)
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error reloading plugin {plugin_name}: {e}")
            return False

    def install_plugin(self, plugin_path: str) -> bool:
        """
        Install a plugin from file or URL

        Args:
            plugin_path: Path to plugin file or URL

        Returns:
            Success status
        """
        try:
            if plugin_path.startswith(('http://', 'https://')):
                # Download plugin
                return self._install_plugin_from_url(plugin_path)
            else:
                # Install from local file
                return self._install_plugin_from_file(plugin_path)

        except Exception as e:
            self.logger.error(f"Error installing plugin from {plugin_path}: {e}")
            return False

    def _install_plugin_from_file(self, file_path: str) -> bool:
        """Install plugin from local file"""
        try:
            if not os.path.exists(file_path):
                return False

            # Determine target directory
            target_dir = self.plugin_directories[0]  # Use first plugin directory

            # Copy file
            filename = os.path.basename(file_path)
            target_path = os.path.join(target_dir, filename)

            import shutil
            shutil.copy2(file_path, target_path)

            # Load the plugin
            self._scan_plugin_directory(target_dir)

            self.logger.info(f"Installed plugin from file: {filename}")
            return True

        except Exception as e:
            self.logger.error(f"Error installing plugin from file: {e}")
            return False

    def _install_plugin_from_url(self, url: str) -> bool:
        """Install plugin from URL"""
        try:
            # Download plugin file
            import requests
            import tempfile

            response = requests.get(url)
            response.raise_for_status()

            # Save to temp file
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.py') as f:
                f.write(response.content)
                temp_path = f.name

            # Install from temp file
            success = self._install_plugin_from_file(temp_path)

            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass

            return success

        except Exception as e:
            self.logger.error(f"Error installing plugin from URL: {e}")
            return False

    def remove_plugin(self, plugin_name: str) -> bool:
        """
        Remove a plugin

        Args:
            plugin_name: Name of plugin to remove

        Returns:
            Success status
        """
        try:
            # Unload plugin
            if not self.unload_plugin(plugin_name):
                return False

            # Remove plugin files
            for directory in self.plugin_directories:
                plugin_file = os.path.join(directory, plugin_name + '.py')
                if os.path.exists(plugin_file):
                    os.remove(plugin_file)

                    # Remove from registry
                    if plugin_name in self.plugins:
                        del self.plugins[plugin_name]

                    self.logger.info(f"Removed plugin: {plugin_name}")
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Error removing plugin {plugin_name}: {e}")
            return False

    def get_plugin_stats(self) -> Dict[str, Any]:
        """Get plugin statistics"""
        enabled_count = sum(1 for plugin in self.plugins.values() if plugin.enabled)
        loaded_count = sum(1 for plugin in self.plugins.values() if plugin.loaded)

        return {
            **self.stats,
            "total_plugins": len(self.plugins),
            "enabled_plugins": enabled_count,
            "loaded_plugins": loaded_count,
            "disabled_plugins": len(self.plugins) - enabled_count,
            "available_hooks": len(self.hooks),
            "total_hooks": sum(len(hooks) for hooks in self.hooks.values())
        }

    def unload_plugins(self):
        """Unload all plugins"""
        try:
            self.logger.info("Unloading all plugins...")

            # Disable all plugins
            for plugin_name in list(self.plugins.keys()):
                self.disable_plugin(plugin_name)

            # Clear loaded plugins
            self.loaded_plugins.clear()

            # Clear hooks
            for hook_name in self.hooks.keys():
                self.hooks[hook_name].clear()

            self.logger.info("All plugins unloaded")

        except Exception as e:
            self.logger.error(f"Error unloading plugins: {e}")

    def create_plugin_skeleton(self, plugin_name: str, target_dir: str = None) -> bool:
        """
        Create a plugin skeleton/template

        Args:
            plugin_name: Name of the plugin
            target_dir: Target directory (uses default if None)

        Returns:
            Success status
        """
        try:
            if not target_dir:
                target_dir = self.plugin_directories[0]

            # Create plugin directory
            plugin_dir = os.path.join(target_dir, plugin_name)
            os.makedirs(plugin_dir, exist_ok=True)

            # Create __init__.py
            init_content = f'''"""
{plugin_name} Plugin for J.A.R.V.I.S.
"""

from jarvis.modules.plugin_manager import Plugin


class JARVISPlugin(Plugin):
    """{plugin_name} plugin class"""

    def __init__(self):
        super().__init__(
            name="{plugin_name}",
            version="1.0.0",
            description="Description of {plugin_name} plugin",
            author="Your Name"
        )

        # Plugin dependencies
        self.dependencies = []

        # Plugin commands
        self.commands = [
            {{
                "name": "example_command",
                "pattern": r"example (.+)",
                "entities": ["parameter"],
                "handler": self.handle_example_command,
                "description": "Example command description"
            }}
        ]

        # Plugin hooks
        self.hooks = {{
            "on_command": self.on_command,
            "on_system_startup": self.on_system_startup,
            "on_system_shutdown": self.on_system_shutdown
        }}

    def handle_example_command(self, command):
        """Handle example command"""
        # Extract entities
        parameter = None
        for entity in command.entities:
            if entity["name"] == "parameter":
                parameter = entity["value"]
                break

        return {{
            "action": "example_command",
            "message": f"Example command executed with parameter: {{parameter}}",
            "data": {{"parameter": parameter}}
        }}

    def on_command(self, command):
        """Called when any command is executed"""
        pass

    def on_system_startup(self):
        """Called when J.A.R.V.I.S. starts up"""
        pass

    def on_system_shutdown(self):
        """Called when J.A.R.V.I.S. shuts down"""
        pass

    def on_enable(self):
        """Called when plugin is enabled"""
        pass

    def on_disable(self):
        """Called when plugin is disabled"""
        pass
'''

            with open(os.path.join(plugin_dir, '__init__.py'), 'w') as f:
                f.write(init_content)

            # Create README.md
            readme_content = f'''# {plugin_name} Plugin

Description of your plugin.

## Installation

1. Copy this directory to your J.A.R.V.I.S. plugins folder
2. Restart J.A.R.V.I.S.
3. Enable the plugin using the plugin manager

## Usage

Describe how to use your plugin.

## Commands

List the commands your plugin provides.

## Configuration

Describe any configuration options.
'''

            with open(os.path.join(plugin_dir, 'README.md'), 'w') as f:
                f.write(readme_content)

            self.logger.info(f"Created plugin skeleton: {plugin_dir}")
            return True

        except Exception as e:
            self.logger.error(f"Error creating plugin skeleton: {e}")
            return False