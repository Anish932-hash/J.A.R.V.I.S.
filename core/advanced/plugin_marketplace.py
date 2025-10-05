"""
J.A.R.V.I.S. Plugin Marketplace
Advanced plugin discovery, installation, management, and marketplace system
"""

import sys
import os
import time
import asyncio
import threading
import uuid
import hashlib
import zipfile
import shutil
from typing import Dict, List, Optional, Any, Set
import logging
import json
import requests
from datetime import datetime, timedelta
import importlib.util
import inspect

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


class PluginMetadata:
    """Plugin metadata container"""

    def __init__(self, metadata_dict: Dict[str, Any]):
        self.id = metadata_dict.get('id', '')
        self.name = metadata_dict.get('name', '')
        self.version = metadata_dict.get('version', '1.0.0')
        self.author = metadata_dict.get('author', 'Unknown')
        self.description = metadata_dict.get('description', '')
        self.category = metadata_dict.get('category', 'general')
        self.tags = metadata_dict.get('tags', [])
        self.dependencies = metadata_dict.get('dependencies', [])
        self.compatibility = metadata_dict.get('compatibility', {})
        self.download_url = metadata_dict.get('download_url', '')
        self.size = metadata_dict.get('size', 0)
        self.hash = metadata_dict.get('hash', '')
        self.rating = metadata_dict.get('rating', 0.0)
        self.download_count = metadata_dict.get('download_count', 0)
        self.last_updated = metadata_dict.get('last_updated', datetime.now().isoformat())
        self.price = metadata_dict.get('price', 0.0)
        self.license = metadata_dict.get('license', 'MIT')

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'version': self.version,
            'author': self.author,
            'description': self.description,
            'category': self.category,
            'tags': self.tags,
            'dependencies': self.dependencies,
            'compatibility': self.compatibility,
            'download_url': self.download_url,
            'size': self.size,
            'hash': self.hash,
            'rating': self.rating,
            'download_count': self.download_count,
            'last_updated': self.last_updated,
            'price': self.price,
            'license': self.license
        }

    def is_compatible(self, jarvis_version: str) -> bool:
        """Check if plugin is compatible with JARVIS version"""
        required_versions = self.compatibility.get('jarvis_versions', [])
        if not required_versions:
            return True

        # Simple version checking (in production, use proper semver)
        return jarvis_version in required_versions or 'all' in required_versions


class InstalledPlugin:
    """Represents an installed plugin"""

    def __init__(self, plugin_id: str, metadata: PluginMetadata, install_path: str):
        self.plugin_id = plugin_id
        self.metadata = metadata
        self.install_path = install_path
        self.installed_at = datetime.now().isoformat()
        self.enabled = True
        self.load_order = 0
        self.dependencies_satisfied = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'plugin_id': self.plugin_id,
            'metadata': self.metadata.to_dict(),
            'install_path': self.install_path,
            'installed_at': self.installed_at,
            'enabled': self.enabled,
            'load_order': self.load_order,
            'dependencies_satisfied': self.dependencies_satisfied
        }


class PluginLoader:
    """Plugin loading and management"""

    def __init__(self):
        self.loaded_plugins = {}
        self.plugin_modules = {}
        self.logger = logging.getLogger('JARVIS.PluginLoader')

    def load_plugin(self, plugin: InstalledPlugin) -> bool:
        """Load a plugin module"""
        try:
            if not plugin.enabled:
                return False

            # Find main plugin file
            main_file = self._find_main_plugin_file(plugin.install_path)
            if not main_file:
                self.logger.error(f"No main plugin file found for {plugin.plugin_id}")
                return False

            # Load module
            spec = importlib.util.spec_from_file_location(plugin.plugin_id, main_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Store loaded module
                self.loaded_plugins[plugin.plugin_id] = plugin
                self.plugin_modules[plugin.plugin_id] = module

                # Initialize plugin if it has init function
                if hasattr(module, 'initialize'):
                    try:
                        asyncio.run(module.initialize())
                    except Exception as e:
                        self.logger.error(f"Error initializing plugin {plugin.plugin_id}: {e}")

                self.logger.info(f"Plugin {plugin.plugin_id} loaded successfully")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error loading plugin {plugin.plugin_id}: {e}")
            return False

    def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin"""
        try:
            if plugin_id in self.plugin_modules:
                module = self.plugin_modules[plugin_id]

                # Shutdown plugin if it has shutdown function
                if hasattr(module, 'shutdown'):
                    try:
                        asyncio.run(module.shutdown())
                    except Exception as e:
                        self.logger.error(f"Error shutting down plugin {plugin_id}: {e}")

                # Remove from loaded plugins
                del self.plugin_modules[plugin_id]
                if plugin_id in self.loaded_plugins:
                    del self.loaded_plugins[plugin_id]

                self.logger.info(f"Plugin {plugin_id} unloaded successfully")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error unloading plugin {plugin_id}: {e}")
            return False

    def _find_main_plugin_file(self, plugin_path: str) -> Optional[str]:
        """Find the main plugin file in the plugin directory"""
        try:
            # Look for common plugin entry points
            candidates = ['__init__.py', 'plugin.py', 'main.py', f"{os.path.basename(plugin_path)}.py"]

            for candidate in candidates:
                candidate_path = os.path.join(plugin_path, candidate)
                if os.path.exists(candidate_path):
                    return candidate_path

            # Look for any .py file
            for file in os.listdir(plugin_path):
                if file.endswith('.py'):
                    return os.path.join(plugin_path, file)

            return None

        except Exception as e:
            self.logger.error(f"Error finding main plugin file: {e}")
            return None

    def get_loaded_plugins(self) -> List[str]:
        """Get list of loaded plugin IDs"""
        return list(self.loaded_plugins.keys())

    def call_plugin_method(self, plugin_id: str, method_name: str, *args, **kwargs) -> Any:
        """Call a method on a loaded plugin"""
        try:
            if plugin_id in self.plugin_modules:
                module = self.plugin_modules[plugin_id]
                if hasattr(module, method_name):
                    method = getattr(module, method_name)
                    return method(*args, **kwargs)

            return None

        except Exception as e:
            self.logger.error(f"Error calling method {method_name} on plugin {plugin_id}: {e}")
            return None


class PluginRepository:
    """Plugin repository management"""

    def __init__(self):
        self.repositories = {}
        self.available_plugins = {}
        self.logger = logging.getLogger('JARVIS.PluginRepository')

    def add_repository(self, repo_url: str, name: str) -> bool:
        """Add a plugin repository"""
        try:
            self.repositories[name] = {
                'url': repo_url,
                'last_updated': None,
                'plugins': []
            }
            self.logger.info(f"Added plugin repository: {name}")
            return True

        except Exception as e:
            self.logger.error(f"Error adding repository {name}: {e}")
            return False

    async def update_repository(self, repo_name: str) -> bool:
        """Update plugin list from repository"""
        try:
            if repo_name not in self.repositories:
                return False

            repo = self.repositories[repo_name]

            if AIOHTTP_AVAILABLE:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{repo['url']}/plugins.json") as response:
                        if response.status == 200:
                            plugin_data = await response.json()
                            repo['plugins'] = plugin_data
                            repo['last_updated'] = datetime.now().isoformat()

                            # Update available plugins
                            for plugin_data in plugin_data:
                                plugin_id = plugin_data['id']
                                self.available_plugins[plugin_id] = PluginMetadata(plugin_data)

                            self.logger.info(f"Updated repository {repo_name}: {len(plugin_data)} plugins")
                            return True
            else:
                # Fallback to requests
                response = requests.get(f"{repo['url']}/plugins.json")
                if response.status_code == 200:
                    plugin_data = response.json()
                    repo['plugins'] = plugin_data
                    repo['last_updated'] = datetime.now().isoformat()

                    for plugin_data in plugin_data:
                        plugin_id = plugin_data['id']
                        self.available_plugins[plugin_id] = PluginMetadata(plugin_data)

                    self.logger.info(f"Updated repository {repo_name}: {len(plugin_data)} plugins")
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Error updating repository {repo_name}: {e}")
            return False

    def search_plugins(self, query: str = '', category: str = '', tags: List[str] = None) -> List[PluginMetadata]:
        """Search available plugins"""
        try:
            results = list(self.available_plugins.values())

            # Filter by query
            if query:
                query_lower = query.lower()
                results = [p for p in results if query_lower in p.name.lower() or query_lower in p.description.lower()]

            # Filter by category
            if category:
                results = [p for p in results if p.category == category]

            # Filter by tags
            if tags:
                results = [p for p in results if any(tag in p.tags for tag in tags)]

            return results

        except Exception as e:
            self.logger.error(f"Error searching plugins: {e}")
            return []

    def get_plugin(self, plugin_id: str) -> Optional[PluginMetadata]:
        """Get plugin metadata by ID"""
        return self.available_plugins.get(plugin_id)


class PluginInstaller:
    """Plugin installation and management"""

    def __init__(self):
        self.install_path = os.path.join(os.path.dirname(__file__), '..', '..', 'plugins')
        self.temp_path = os.path.join(self.install_path, 'temp')
        self.logger = logging.getLogger('JARVIS.PluginInstaller')

        # Ensure directories exist
        os.makedirs(self.install_path, exist_ok=True)
        os.makedirs(self.temp_path, exist_ok=True)

    async def install_plugin(self, plugin: PluginMetadata, progress_callback: callable = None) -> Dict[str, Any]:
        """Install a plugin"""
        try:
            install_id = str(uuid.uuid4())[:8]

            # Create installation directory
            plugin_dir = os.path.join(self.install_path, plugin.id)
            os.makedirs(plugin_dir, exist_ok=True)

            # Download plugin
            download_result = await self._download_plugin(plugin, plugin_dir, progress_callback)
            if not download_result['success']:
                return download_result

            # Verify integrity
            if not self._verify_plugin_integrity(plugin_dir, plugin.hash):
                return {
                    'success': False,
                    'error': 'Plugin integrity verification failed',
                    'install_id': install_id
                }

            # Extract plugin
            extract_result = self._extract_plugin(plugin_dir, download_result['archive_path'])
            if not extract_result['success']:
                return extract_result

            # Install dependencies
            dep_result = await self._install_dependencies(plugin.dependencies)
            if not dep_result['success']:
                return dep_result

            # Create installed plugin record
            installed_plugin = InstalledPlugin(plugin.id, plugin, plugin_dir)

            # Save installation metadata
            self._save_installation_metadata(installed_plugin)

            self.logger.info(f"Plugin {plugin.name} installed successfully")
            return {
                'success': True,
                'install_id': install_id,
                'plugin': installed_plugin.to_dict(),
                'message': f"Plugin {plugin.name} installed successfully"
            }

        except Exception as e:
            self.logger.error(f"Error installing plugin {plugin.name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'install_id': install_id if 'install_id' in locals() else None
            }

    async def _download_plugin(self, plugin: PluginMetadata, target_dir: str, progress_callback: callable = None) -> Dict[str, Any]:
        """Download plugin archive"""
        try:
            if not plugin.download_url:
                return {'success': False, 'error': 'No download URL provided'}

            archive_path = os.path.join(target_dir, f"{plugin.id}.zip")

            if AIOHTTP_AVAILABLE:
                async with aiohttp.ClientSession() as session:
                    async with session.get(plugin.download_url) as response:
                        if response.status != 200:
                            return {'success': False, 'error': f'Download failed: HTTP {response.status}'}

                        total_size = int(response.headers.get('Content-Length', 0))
                        downloaded = 0

                        with open(archive_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                                downloaded += len(chunk)

                                if progress_callback and total_size > 0:
                                    progress = (downloaded / total_size) * 100
                                    progress_callback(progress)
            else:
                # Fallback to requests
                response = requests.get(plugin.download_url, stream=True)
                if response.status_code != 200:
                    return {'success': False, 'error': f'Download failed: HTTP {response.status_code}'}

                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0

                with open(archive_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)

                        if progress_callback and total_size > 0:
                            progress = (downloaded / total_size) * 100
                            progress_callback(progress)

            return {
                'success': True,
                'archive_path': archive_path,
                'size': downloaded
            }

        except Exception as e:
            self.logger.error(f"Error downloading plugin: {e}")
            return {'success': False, 'error': str(e)}

    def _extract_plugin(self, plugin_dir: str, archive_path: str) -> Dict[str, Any]:
        """Extract plugin archive"""
        try:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(plugin_dir)

            # Remove archive
            os.remove(archive_path)

            return {'success': True}

        except Exception as e:
            self.logger.error(f"Error extracting plugin: {e}")
            return {'success': False, 'error': str(e)}

    def _verify_plugin_integrity(self, plugin_dir: str, expected_hash: str) -> bool:
        """Verify plugin integrity using hash"""
        try:
            if not expected_hash:
                return True  # No hash to verify

            # Calculate directory hash
            calculated_hash = self._calculate_directory_hash(plugin_dir)

            return calculated_hash == expected_hash

        except Exception as e:
            self.logger.error(f"Error verifying plugin integrity: {e}")
            return False

    def _calculate_directory_hash(self, directory: str) -> str:
        """Calculate SHA256 hash of all files in directory"""
        try:
            sha256 = hashlib.sha256()

            for root, dirs, files in os.walk(directory):
                for file in sorted(files):  # Sort for consistent hashing
                    file_path = os.path.join(root, file)
                    with open(file_path, 'rb') as f:
                        while chunk := f.read(8192):
                            sha256.update(chunk)

            return sha256.hexdigest()

        except Exception as e:
            self.logger.error(f"Error calculating directory hash: {e}")
            return ""

    async def _install_dependencies(self, dependencies: List[str]) -> Dict[str, Any]:
        """Install plugin dependencies"""
        try:
            if not dependencies:
                return {'success': True}

            # In a real implementation, this would use pip or similar
            # For now, simulate dependency installation
            self.logger.info(f"Installing dependencies: {dependencies}")

            # Simulate pip install
            for dep in dependencies:
                self.logger.info(f"Installing {dep}...")
                # subprocess.run(['pip', 'install', dep])  # Would be used in real implementation

            return {'success': True, 'installed': dependencies}

        except Exception as e:
            self.logger.error(f"Error installing dependencies: {e}")
            return {'success': False, 'error': str(e)}

    def _save_installation_metadata(self, plugin: InstalledPlugin):
        """Save plugin installation metadata"""
        try:
            metadata_file = os.path.join(plugin.install_path, 'plugin.json')
            with open(metadata_file, 'w') as f:
                json.dump(plugin.to_dict(), f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving installation metadata: {e}")

    def uninstall_plugin(self, plugin_id: str) -> bool:
        """Uninstall a plugin"""
        try:
            plugin_dir = os.path.join(self.install_path, plugin_id)

            if os.path.exists(plugin_dir):
                shutil.rmtree(plugin_dir)
                self.logger.info(f"Plugin {plugin_id} uninstalled successfully")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error uninstalling plugin {plugin_id}: {e}")
            return False


class PluginMarketplace:
    """Advanced plugin marketplace system"""

    def __init__(self, development_engine):
        self.development_engine = development_engine
        self.jarvis = development_engine.jarvis if hasattr(development_engine, 'jarvis') else None
        self.logger = logging.getLogger('JARVIS.PluginMarketplace')

        # Marketplace components
        self.repository = PluginRepository()
        self.installer = PluginInstaller()
        self.loader = PluginLoader()

        # Plugin management
        self.installed_plugins = {}
        self.jarvis_version = "2.0.0"  # Would be dynamic in real implementation

        # Update management
        self.update_check_interval = 3600  # 1 hour
        self.last_update_check = None

    async def initialize(self):
        """Initialize plugin marketplace"""
        try:
            self.logger.info("Initializing Plugin Marketplace...")

            # Add default repositories
            await self._add_default_repositories()

            # Load installed plugins
            await self._load_installed_plugins()

            # Load plugins
            await self._load_plugins()

            self.logger.info("Plugin Marketplace initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing plugin marketplace: {e}")
            return False

    async def _add_default_repositories(self):
        """Add default plugin repositories"""
        try:
            # Add official JARVIS repository
            self.repository.add_repository("https://plugins.jarvis.ai", "official")

            # Add community repository
            self.repository.add_repository("https://community.jarvis.ai/plugins", "community")

            # Update repositories
            for repo_name in self.repository.repositories.keys():
                await self.repository.update_repository(repo_name)

        except Exception as e:
            self.logger.error(f"Error adding default repositories: {e}")

    async def _load_installed_plugins(self):
        """Load installed plugins from disk"""
        try:
            for plugin_dir in os.listdir(self.installer.install_path):
                plugin_path = os.path.join(self.installer.install_path, plugin_dir)
                if os.path.isdir(plugin_path):
                    metadata_file = os.path.join(plugin_path, 'plugin.json')
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            plugin_data = json.load(f)

                        plugin_id = plugin_data['plugin_id']
                        metadata = PluginMetadata(plugin_data['metadata'])
                        installed_plugin = InstalledPlugin(plugin_id, metadata, plugin_path)

                        # Restore state
                        installed_plugin.enabled = plugin_data.get('enabled', True)
                        installed_plugin.load_order = plugin_data.get('load_order', 0)
                        installed_plugin.dependencies_satisfied = plugin_data.get('dependencies_satisfied', True)

                        self.installed_plugins[plugin_id] = installed_plugin

            self.logger.info(f"Loaded {len(self.installed_plugins)} installed plugins")

        except Exception as e:
            self.logger.error(f"Error loading installed plugins: {e}")

    async def _load_plugins(self):
        """Load all enabled plugins"""
        try:
            # Sort by load order
            sorted_plugins = sorted(
                self.installed_plugins.values(),
                key=lambda p: p.load_order
            )

            loaded_count = 0
            for plugin in sorted_plugins:
                if plugin.enabled and plugin.dependencies_satisfied:
                    if self.loader.load_plugin(plugin):
                        loaded_count += 1

            self.logger.info(f"Loaded {loaded_count} plugins")

        except Exception as e:
            self.logger.error(f"Error loading plugins: {e}")

    async def search_plugins(self, query: str = '', category: str = '', tags: List[str] = None) -> List[Dict[str, Any]]:
        """Search available plugins"""
        try:
            plugins = self.repository.search_plugins(query, category, tags or [])

            # Add installation status
            results = []
            for plugin in plugins:
                if plugin.is_compatible(self.jarvis_version):
                    result = plugin.to_dict()
                    result['installed'] = plugin.id in self.installed_plugins
                    result['enabled'] = self.installed_plugins.get(plugin.id, InstalledPlugin('', plugin, '')).enabled
                    results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"Error searching plugins: {e}")
            return []

    async def install_plugin(self, plugin_id: str, progress_callback: callable = None) -> Dict[str, Any]:
        """Install a plugin"""
        try:
            plugin = self.repository.get_plugin(plugin_id)
            if not plugin:
                return {'success': False, 'error': 'Plugin not found'}

            if not plugin.is_compatible(self.jarvis_version):
                return {'success': False, 'error': 'Plugin not compatible with current JARVIS version'}

            if plugin_id in self.installed_plugins:
                return {'success': False, 'error': 'Plugin already installed'}

            # Install plugin
            result = await self.installer.install_plugin(plugin, progress_callback)

            if result['success']:
                # Add to installed plugins
                installed_plugin = InstalledPlugin(
                    plugin_id,
                    plugin,
                    os.path.join(self.installer.install_path, plugin_id)
                )
                self.installed_plugins[plugin_id] = installed_plugin

                # Load plugin
                if self.loader.load_plugin(installed_plugin):
                    result['loaded'] = True
                else:
                    result['loaded'] = False
                    result['warning'] = 'Plugin installed but failed to load'

            return result

        except Exception as e:
            self.logger.error(f"Error installing plugin {plugin_id}: {e}")
            return {'success': False, 'error': str(e)}

    async def uninstall_plugin(self, plugin_id: str) -> Dict[str, Any]:
        """Uninstall a plugin"""
        try:
            if plugin_id not in self.installed_plugins:
                return {'success': False, 'error': 'Plugin not installed'}

            # Unload plugin
            self.loader.unload_plugin(plugin_id)

            # Uninstall plugin
            if self.installer.uninstall_plugin(plugin_id):
                del self.installed_plugins[plugin_id]
                return {'success': True, 'message': f'Plugin {plugin_id} uninstalled successfully'}

            return {'success': False, 'error': 'Failed to uninstall plugin'}

        except Exception as e:
            self.logger.error(f"Error uninstalling plugin {plugin_id}: {e}")
            return {'success': False, 'error': str(e)}

    async def update_plugins(self) -> Dict[str, Any]:
        """Update all installed plugins"""
        try:
            updated = []
            failed = []

            for plugin_id, installed_plugin in self.installed_plugins.items():
                try:
                    # Check if update is available
                    latest_plugin = self.repository.get_plugin(plugin_id)
                    if latest_plugin and latest_plugin.version != installed_plugin.metadata.version:
                        # Update plugin
                        self.logger.info(f"Updating plugin {plugin_id} from {installed_plugin.metadata.version} to {latest_plugin.version}")

                        # Unload current version
                        self.loader.unload_plugin(plugin_id)

                        # Install new version
                        update_result = await self.installer.install_plugin(latest_plugin)
                        if update_result['success']:
                            # Update installed plugin record
                            installed_plugin.metadata = latest_plugin
                            self._save_plugin_metadata(installed_plugin)

                            # Reload plugin
                            self.loader.load_plugin(installed_plugin)

                            updated.append(plugin_id)
                        else:
                            failed.append(f"{plugin_id}: {update_result.get('error', 'Unknown error')}")

                except Exception as e:
                    failed.append(f"{plugin_id}: {str(e)}")

            return {
                'success': True,
                'updated': updated,
                'failed': failed,
                'total_updated': len(updated),
                'total_failed': len(failed)
            }

        except Exception as e:
            self.logger.error(f"Error updating plugins: {e}")
            return {'success': False, 'error': str(e)}

    def _save_plugin_metadata(self, plugin: InstalledPlugin):
        """Save plugin metadata"""
        try:
            metadata_file = os.path.join(plugin.install_path, 'plugin.json')
            with open(metadata_file, 'w') as f:
                json.dump(plugin.to_dict(), f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving plugin metadata: {e}")

    def enable_plugin(self, plugin_id: str) -> bool:
        """Enable a plugin"""
        try:
            if plugin_id in self.installed_plugins:
                plugin = self.installed_plugins[plugin_id]
                plugin.enabled = True
                self._save_plugin_metadata(plugin)

                # Load plugin
                self.loader.load_plugin(plugin)
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error enabling plugin {plugin_id}: {e}")
            return False

    def disable_plugin(self, plugin_id: str) -> bool:
        """Disable a plugin"""
        try:
            if plugin_id in self.installed_plugins:
                plugin = self.installed_plugins[plugin_id]
                plugin.enabled = False
                self._save_plugin_metadata(plugin)

                # Unload plugin
                self.loader.unload_plugin(plugin_id)
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error disabling plugin {plugin_id}: {e}")
            return False

    def get_installed_plugins(self) -> List[Dict[str, Any]]:
        """Get list of installed plugins"""
        try:
            return [plugin.to_dict() for plugin in self.installed_plugins.values()]

        except Exception as e:
            self.logger.error(f"Error getting installed plugins: {e}")
            return []

    def get_plugin_info(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed plugin information"""
        try:
            if plugin_id in self.installed_plugins:
                plugin = self.installed_plugins[plugin_id]
                info = plugin.to_dict()
                info['loaded'] = plugin_id in self.loader.loaded_plugins
                return info

            return None

        except Exception as e:
            self.logger.error(f"Error getting plugin info for {plugin_id}: {e}")
            return None

    def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get marketplace statistics"""
        try:
            installed_count = len(self.installed_plugins)
            loaded_count = len(self.loader.loaded_plugins)
            available_count = len(self.repository.available_plugins)

            enabled_count = sum(1 for p in self.installed_plugins.values() if p.enabled)
            disabled_count = installed_count - enabled_count

            return {
                'installed_plugins': installed_count,
                'loaded_plugins': loaded_count,
                'available_plugins': available_count,
                'enabled_plugins': enabled_count,
                'disabled_plugins': disabled_count,
                'repositories': len(self.repository.repositories),
                'last_update_check': self.last_update_check
            }

        except Exception as e:
            self.logger.error(f"Error getting marketplace stats: {e}")
            return {}

    def call_plugin_method(self, plugin_id: str, method_name: str, *args, **kwargs) -> Any:
        """Call a method on a loaded plugin"""
        return self.loader.call_plugin_method(plugin_id, method_name, *args, **kwargs)

    async def check_for_updates(self) -> Dict[str, Any]:
        """Check for plugin updates"""
        try:
            current_time = datetime.now()

            # Check if we should update
            if self.last_update_check and (current_time - datetime.fromisoformat(self.last_update_check)).seconds < self.update_check_interval:
                return {'success': True, 'message': 'Update check skipped - too soon'}

            # Update repositories
            updated_repos = 0
            for repo_name in self.repository.repositories.keys():
                if await self.repository.update_repository(repo_name):
                    updated_repos += 1

            self.last_update_check = current_time.isoformat()

            # Check for updates
            updates_available = []
            for plugin_id, installed_plugin in self.installed_plugins.items():
                latest_plugin = self.repository.get_plugin(plugin_id)
                if latest_plugin and latest_plugin.version != installed_plugin.metadata.version:
                    updates_available.append({
                        'plugin_id': plugin_id,
                        'current_version': installed_plugin.metadata.version,
                        'latest_version': latest_plugin.version,
                        'name': latest_plugin.name
                    })

            return {
                'success': True,
                'repositories_updated': updated_repos,
                'updates_available': updates_available,
                'total_updates': len(updates_available)
            }

        except Exception as e:
            self.logger.error(f"Error checking for updates: {e}")
            return {'success': False, 'error': str(e)}

    async def shutdown(self):
        """Shutdown plugin marketplace"""
        try:
            # Unload all plugins
            for plugin_id in list(self.loader.loaded_plugins.keys()):
                self.loader.unload_plugin(plugin_id)

            self.logger.info("Plugin Marketplace shutdown")
        except Exception as e:
            self.logger.error(f"Error shutting down plugin marketplace: {e}")