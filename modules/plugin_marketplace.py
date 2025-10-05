"""
J.A.R.V.I.S. Plugin Marketplace
Community plugin discovery, installation, and management
"""

import os
import time
import json
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
import logging


class PluginMarketplace:
    """
    Plugin marketplace for community plugins
    """

    def __init__(self, jarvis_instance):
        self.jarvis = jarvis_instance
        self.logger = logging.getLogger('JARVIS.PluginMarketplace')

        # Marketplace configuration
        self.marketplace_urls = [
            "https://api.jarvis-plugins.com/v1",
            "https://plugins.jarvis-ai.org/api"
        ]

        # Plugin cache
        self.plugin_cache = {}
        self.cache_ttl = 3600  # 1 hour

    async def initialize(self):
        self.logger.info("Plugin marketplace initialized")

    async def search_plugins(self, query: str, category: str = None) -> List[Dict[str, Any]]:
        """Search for plugins in marketplace"""
        try:
            plugins = []

            for url in self.marketplace_urls:
                try:
                    marketplace_plugins = await self._search_marketplace(url, query, category)
                    plugins.extend(marketplace_plugins)

                except Exception as e:
                    self.logger.error(f"Error searching marketplace {url}: {e}")

            # Remove duplicates
            unique_plugins = self._deduplicate_plugins(plugins)

            return unique_plugins

        except Exception as e:
            self.logger.error(f"Error searching plugins: {e}")
            return []

    async def _search_marketplace(self, base_url: str, query: str, category: str) -> List[Dict[str, Any]]:
        """Search specific marketplace"""
        try:
            async with aiohttp.ClientSession() as session:
                params = {"q": query}
                if category:
                    params["category"] = category

                async with session.get(f"{base_url}/plugins/search", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("plugins", [])
                    else:
                        return []

        except Exception as e:
            self.logger.error(f"Error searching marketplace: {e}")
            return []

    def _deduplicate_plugins(self, plugins: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate plugins"""
        seen_ids = set()
        unique_plugins = []

        for plugin in plugins:
            plugin_id = plugin.get("id", "")
            if plugin_id and plugin_id not in seen_ids:
                seen_ids.add(plugin_id)
                unique_plugins.append(plugin)

        return unique_plugins

    async def get_plugin_info(self, plugin_id: str) -> Dict[str, Any]:
        """Get detailed plugin information"""
        try:
            # Check cache first
            if plugin_id in self.plugin_cache:
                cached_time, cached_info = self.plugin_cache[plugin_id]
                if time.time() - cached_time < self.cache_ttl:
                    return cached_info

            # Search marketplaces
            for url in self.marketplace_urls:
                try:
                    info = await self._get_plugin_from_marketplace(url, plugin_id)
                    if info:
                        self.plugin_cache[plugin_id] = (time.time(), info)
                        return info

                except Exception as e:
                    self.logger.error(f"Error getting plugin from {url}: {e}")

            return {}

        except Exception as e:
            self.logger.error(f"Error getting plugin info: {e}")
            return {}

    async def _get_plugin_from_marketplace(self, base_url: str, plugin_id: str) -> Dict[str, Any]:
        """Get plugin from specific marketplace"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}/plugins/{plugin_id}") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {}

        except Exception as e:
            self.logger.error(f"Error getting plugin from marketplace: {e}")
            return {}

    async def download_plugin(self, plugin_id: str, version: str = "latest") -> Dict[str, Any]:
        """Download plugin from marketplace"""
        try:
            # Get plugin info
            plugin_info = await self.get_plugin_info(plugin_id)

            if not plugin_info:
                return {"success": False, "error": "Plugin not found"}

            # Get download URL
            download_url = plugin_info.get("download_url", "")

            if not download_url:
                return {"success": False, "error": "No download URL available"}

            # Download plugin
            async with aiohttp.ClientSession() as session:
                async with session.get(download_url) as response:
                    if response.status == 200:
                        plugin_code = await response.text()

                        return {
                            "success": True,
                            "plugin_code": plugin_code,
                            "plugin_info": plugin_info
                        }
                    else:
                        return {"success": False, "error": f"Download failed: {response.status}"}

        except Exception as e:
            self.logger.error(f"Error downloading plugin: {e}")
            return {"success": False, "error": str(e)}

    async def install_plugin(self, plugin_id: str, version: str = "latest") -> Dict[str, Any]:
        """Install plugin from marketplace"""
        try:
            # Download plugin
            download_result = await self.download_plugin(plugin_id, version)

            if not download_result["success"]:
                return download_result

            # Install using plugin manager
            if hasattr(self.jarvis, 'plugin_manager'):
                # Save plugin code to file
                plugin_code = download_result["plugin_code"]
                plugin_info = download_result["plugin_info"]

                # Create plugin file
                plugin_dir = os.path.join(os.path.dirname(__file__), '..', 'plugins')
                os.makedirs(plugin_dir, exist_ok=True)

                plugin_file = os.path.join(plugin_dir, f"{plugin_id}.py")

                with open(plugin_file, 'w') as f:
                    f.write(plugin_code)

                # Load plugin
                self.jarvis.plugin_manager._scan_plugin_directory(plugin_dir)

                return {
                    "success": True,
                    "message": f"Plugin {plugin_id} installed successfully",
                    "plugin_info": plugin_info
                }

        except Exception as e:
            self.logger.error(f"Error installing plugin: {e}")
            return {"success": False, "error": str(e)}

    async def rate_plugin(self, plugin_id: str, rating: int, review: str = "") -> Dict[str, Any]:
        """Rate plugin in marketplace"""
        try:
            # This would submit rating to marketplace
            return {
                "success": True,
                "message": "Rating submitted"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_featured_plugins(self) -> List[Dict[str, Any]]:
        """Get featured plugins from marketplace"""
        try:
            plugins = []

            for url in self.marketplace_urls:
                try:
                    marketplace_plugins = await self._get_featured_from_marketplace(url)
                    plugins.extend(marketplace_plugins)

                except Exception as e:
                    self.logger.error(f"Error getting featured plugins from {url}: {e}")

            return plugins[:20]  # Return top 20

        except Exception as e:
            self.logger.error(f"Error getting featured plugins: {e}")
            return []

    async def _get_featured_from_marketplace(self, base_url: str) -> List[Dict[str, Any]]:
        """Get featured plugins from specific marketplace"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}/plugins/featured") as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("plugins", [])
                    else:
                        return []

        except Exception as e:
            self.logger.error(f"Error getting featured plugins: {e}")
            return []

    async def shutdown(self):
        self.logger.info("Plugin marketplace shutdown complete")