"""
J.A.R.V.I.S. Auto-Updater System
Handles automatic updates, patches, and version management
"""

import os
import sys
import time
import json
import hashlib
import asyncio
import zipfile
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import logging
import urllib.request
import urllib.error
import requests

class UpdateManager:
    """
    Advanced auto-updater system for J.A.R.V.I.S.
    Handles version checking, downloading, and installation of updates.
    """

    def __init__(self, jarvis_instance):
        """
        Initialize update manager

        Args:
            jarvis_instance: Reference to main JARVIS instance
        """
        self.jarvis = jarvis_instance
        self.logger = logging.getLogger('JARVIS.Updater')

        # Update configuration
        self.update_config = {
            "update_url": "https://api.jarvis.ai/updates/",
            "check_interval": 86400,  # 24 hours
            "auto_update": True,
            "beta_updates": False,
            "backup_before_update": True,
            "update_channel": "stable"
        }

        # Update state
        self.current_version = "2.0.0"
        self.latest_version = None
        self.update_available = False
        self.download_progress = 0
        self.installing = False

        # Update history
        self.update_history: List[Dict[str, Any]] = []

        # Callbacks
        self.update_callbacks: List[Callable] = []

    async def initialize(self):
        """Initialize update manager"""
        try:
            self.logger.info("Initializing update manager...")

            # Load update configuration
            await self._load_config()

            # Load update history
            await self._load_history()

            # Start background update checking
            if self.update_config["auto_update"]:
                asyncio.create_task(self._start_auto_check())

            self.logger.info("Update manager initialized")

        except Exception as e:
            self.logger.error(f"Error initializing update manager: {e}")
            raise

    async def check_for_updates(self, force: bool = False) -> Dict[str, Any]:
        """
        Check for available updates

        Args:
            force: Force check even if recently checked

        Returns:
            Update check result
        """
        try:
            self.logger.info("Checking for updates...")

            # Check if we should skip (recently checked)
            if not force and not self._should_check_updates():
                return {
                    "update_available": False,
                    "message": "Recently checked, skipping"
                }

            # Get latest version info
            version_info = await self._fetch_version_info()

            if not version_info:
                return {
                    "update_available": False,
                    "error": "Failed to fetch version info"
                }

            self.latest_version = version_info.get("version")
            self.update_available = self._compare_versions(self.current_version, self.latest_version)

            result = {
                "update_available": self.update_available,
                "current_version": self.current_version,
                "latest_version": self.latest_version,
                "changelog": version_info.get("changelog", []),
                "download_url": version_info.get("download_url"),
                "release_date": version_info.get("release_date")
            }

            if self.update_available:
                self.logger.info(f"Update available: {self.latest_version}")
                # Notify callbacks
                await self._notify_update_available(result)
            else:
                self.logger.info("No updates available")

            # Save last check time
            await self._save_last_check()

            return result

        except Exception as e:
            self.logger.error(f"Error checking for updates: {e}")
            return {
                "update_available": False,
                "error": str(e)
            }

    async def download_update(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Download available update

        Args:
            progress_callback: Callback for download progress

        Returns:
            Download result
        """
        try:
            if not self.update_available or not self.latest_version:
                return {
                    "success": False,
                    "error": "No update available"
                }

            self.logger.info(f"Downloading update: {self.latest_version}")

            # Get download URL
            version_info = await self._fetch_version_info()
            download_url = version_info.get("download_url")

            if not download_url:
                return {
                    "success": False,
                    "error": "No download URL available"
                }

            # Create temp directory
            temp_dir = Path(tempfile.gettempdir()) / "jarvis_update"
            temp_dir.mkdir(exist_ok=True)

            # Download file
            update_file = temp_dir / f"jarvis_update_{self.latest_version}.zip"

            success = await self._download_file(
                download_url,
                update_file,
                progress_callback
            )

            if not success:
                return {
                    "success": False,
                    "error": "Download failed"
                }

            # Verify download
            if not await self._verify_download(update_file, version_info):
                return {
                    "success": False,
                    "error": "Download verification failed"
                }

            return {
                "success": True,
                "update_file": str(update_file),
                "version": self.latest_version
            }

        except Exception as e:
            self.logger.error(f"Error downloading update: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def install_update(self, update_file: str, backup: bool = True) -> Dict[str, Any]:
        """
        Install downloaded update

        Args:
            update_file: Path to update file
            backup: Whether to backup current installation

        Returns:
            Installation result
        """
        try:
            self.installing = True
            self.logger.info(f"Installing update from: {update_file}")

            update_path = Path(update_file)
            if not update_path.exists():
                return {
                    "success": False,
                    "error": "Update file not found"
                }

            # Create backup if requested
            if backup and self.update_config["backup_before_update"]:
                backup_result = await self._create_backup()
                if not backup_result["success"]:
                    self.logger.warning("Backup creation failed, continuing with update")

            # Extract update
            extract_result = await self._extract_update(update_path)
            if not extract_result["success"]:
                return extract_result

            # Apply update
            apply_result = await self._apply_update(extract_result["extract_path"])
            if not apply_result["success"]:
                # Try to restore backup
                if backup:
                    await self._restore_backup()
                return apply_result

            # Update version
            self.current_version = self.latest_version
            self.update_available = False

            # Record update in history
            await self._record_update(self.latest_version, "successful")

            # Cleanup
            await self._cleanup_temp_files()

            # Restart application if needed
            if apply_result.get("restart_required", False):
                await self._schedule_restart()

            self.installing = False

            result = {
                "success": True,
                "version": self.current_version,
                "restart_required": apply_result.get("restart_required", False),
                "message": "Update installed successfully"
            }

            # Notify callbacks
            await self._notify_update_installed(result)

            return result

        except Exception as e:
            self.logger.error(f"Error installing update: {e}")
            self.installing = False
            return {
                "success": False,
                "error": str(e)
            }

    async def _start_auto_check(self):
        """Start automatic update checking"""
        while True:
            try:
                await self.check_for_updates()
                await asyncio.sleep(self.update_config["check_interval"])
            except Exception as e:
                self.logger.error(f"Error in auto update check: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour

    def _should_check_updates(self) -> bool:
        """Check if we should perform update check"""
        try:
            last_check_file = Path(os.path.dirname(__file__)) / ".." / ".." / "data" / "last_update_check.json"

            if not last_check_file.exists():
                return True

            with open(last_check_file, 'r') as f:
                data = json.load(f)

            last_check = data.get("last_check", 0)
            return (time.time() - last_check) > self.update_config["check_interval"]

        except Exception:
            return True

    async def _fetch_version_info(self) -> Optional[Dict[str, Any]]:
        """Fetch version information from update server"""
        try:
            url = f"{self.update_config['update_url']}version.json"

            # Make HTTP request to fetch version info
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            version_data = response.json()

            # Validate required fields
            required_fields = ["version", "download_url"]
            for field in required_fields:
                if field not in version_data:
                    self.logger.error(f"Missing required field in version info: {field}")
                    return None

            self.logger.info(f"Fetched version info: {version_data.get('version')}")
            return version_data

        except requests.exceptions.RequestException as e:
            self.logger.error(f"HTTP error fetching version info: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in version info response: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching version info: {e}")
            return None

    def _compare_versions(self, current: str, latest: str) -> bool:
        """Compare version strings"""
        try:
            current_parts = [int(x) for x in current.split('.')]
            latest_parts = [int(x) for x in latest.split('.')]

            return latest_parts > current_parts

        except Exception:
            return False

    async def _download_file(self, url: str, dest_path: Path,
                           progress_callback: Optional[Callable] = None) -> bool:
        """Download file with progress tracking"""
        try:
            self.logger.info(f"Downloading from: {url}")

            # Create request
            req = urllib.request.Request(url)

            # Download with progress
            with urllib.request.urlopen(req) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0

                with open(dest_path, 'wb') as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break

                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            self.download_progress = progress

                            if progress_callback:
                                progress_callback(progress)

            self.logger.info(f"Download completed: {dest_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error downloading file: {e}")
            return False

    async def _verify_download(self, file_path: Path, version_info: Dict[str, Any]) -> bool:
        """Verify downloaded file integrity"""
        try:
            expected_checksum = version_info.get("checksum")
            if not expected_checksum:
                self.logger.warning("No checksum available for verification")
                return True

            # Calculate file checksum
            hash_md5 = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)

            actual_checksum = hash_md5.hexdigest()

            if actual_checksum == expected_checksum:
                self.logger.info("Download verification successful")
                return True
            else:
                self.logger.error(f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}")
                return False

        except Exception as e:
            self.logger.error(f"Error verifying download: {e}")
            return False

    async def _create_backup(self) -> Dict[str, Any]:
        """Create backup of current installation"""
        try:
            self.logger.info("Creating backup...")

            jarvis_root = Path(os.path.dirname(__file__)) / ".." / ".."
            backup_dir = Path(tempfile.gettempdir()) / f"jarvis_backup_{int(time.time())}"
            backup_dir.mkdir(exist_ok=True)

            # Files to backup
            backup_files = [
                "config/jarvis.json",
                "data/",
                "plugins/",
                "logs/"
            ]

            for file_path in backup_files:
                src_path = jarvis_root / file_path
                if src_path.exists():
                    if src_path.is_file():
                        shutil.copy2(src_path, backup_dir / Path(file_path).name)
                    else:
                        shutil.copytree(src_path, backup_dir / Path(file_path).stem, dirs_exist_ok=True)

            # Save backup info
            backup_info = {
                "timestamp": time.time(),
                "version": self.current_version,
                "backup_path": str(backup_dir)
            }

            with open(backup_dir / "backup_info.json", 'w') as f:
                json.dump(backup_info, f, indent=2)

            self.logger.info(f"Backup created: {backup_dir}")
            return {
                "success": True,
                "backup_path": str(backup_dir)
            }

        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _restore_backup(self):
        """Restore from backup"""
        try:
            self.logger.info("Restoring from backup...")

            # Find latest backup
            temp_dir = Path(tempfile.gettempdir())
            backup_dirs = [d for d in temp_dir.iterdir() if d.name.startswith("jarvis_backup_")]

            if not backup_dirs:
                self.logger.error("No backup found")
                return False

            latest_backup = max(backup_dirs, key=lambda x: x.stat().st_mtime)

            # Restore files
            jarvis_root = Path(os.path.dirname(__file__)) / ".." / ".."

            for item in latest_backup.iterdir():
                if item.name == "backup_info.json":
                    continue

                dest_path = jarvis_root / item.name
                if item.is_file():
                    shutil.copy2(item, dest_path)
                else:
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    shutil.copytree(item, dest_path)

            self.logger.info("Backup restored successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error restoring backup: {e}")
            return False

    async def _extract_update(self, update_file: Path) -> Dict[str, Any]:
        """Extract update archive"""
        try:
            self.logger.info(f"Extracting update: {update_file}")

            extract_path = Path(tempfile.gettempdir()) / f"jarvis_update_extracted_{int(time.time())}"
            extract_path.mkdir(exist_ok=True)

            with zipfile.ZipFile(update_file, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

            self.logger.info(f"Update extracted to: {extract_path}")
            return {
                "success": True,
                "extract_path": str(extract_path)
            }

        except Exception as e:
            self.logger.error(f"Error extracting update: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _apply_update(self, extract_path: str) -> Dict[str, Any]:
        """Apply extracted update files"""
        try:
            self.logger.info("Applying update...")

            extract_path = Path(extract_path)
            jarvis_root = Path(os.path.dirname(__file__)) / ".." / ".."

            # Update manifest
            manifest_file = extract_path / "update_manifest.json"
            if manifest_file.exists():
                with open(manifest_file, 'r') as f:
                    manifest = json.load(f)

                # Apply file updates
                for file_update in manifest.get("file_updates", []):
                    src = extract_path / file_update["source"]
                    dest = jarvis_root / file_update["destination"]

                    dest.parent.mkdir(parents=True, exist_ok=True)

                    if src.is_file():
                        shutil.copy2(src, dest)
                    elif src.is_dir():
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(src, dest)

                restart_required = manifest.get("restart_required", True)

            else:
                # Simple file replacement
                restart_required = True

                # Copy all files from extract path
                for item in extract_path.iterdir():
                    if item.name == "update_manifest.json":
                        continue

                    dest = jarvis_root / item.name
                    if item.is_file():
                        shutil.copy2(item, dest)
                    elif item.is_dir():
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(item, dest)

            self.logger.info("Update applied successfully")
            return {
                "success": True,
                "restart_required": restart_required
            }

        except Exception as e:
            self.logger.error(f"Error applying update: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _schedule_restart(self):
        """Schedule application restart"""
        try:
            self.logger.info("Scheduling application restart...")

            # Create restart script
            restart_script = f'''
import sys
import os
import time

# Wait a moment for current process to exit
time.sleep(2)

# Restart application
os.execv(sys.executable, [sys.executable] + sys.argv)
'''

            script_path = Path(tempfile.gettempdir()) / "jarvis_restart.py"
            with open(script_path, 'w') as f:
                f.write(restart_script)

            # Launch restart script
            subprocess.Popen([sys.executable, str(script_path)], creationflags=subprocess.CREATE_NEW_CONSOLE)

        except Exception as e:
            self.logger.error(f"Error scheduling restart: {e}")

    async def _save_last_check(self):
        """Save last update check timestamp"""
        try:
            data_dir = Path(os.path.dirname(__file__)) / ".." / ".." / "data"
            data_dir.mkdir(exist_ok=True)

            data = {
                "last_check": time.time(),
                "current_version": self.current_version
            }

            with open(data_dir / "last_update_check.json", 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving last check: {e}")

    async def _load_config(self):
        """Load update configuration"""
        try:
            config_file = Path(os.path.dirname(__file__)) / ".." / ".." / "config" / "updater.json"

            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    self.update_config.update(config)

        except Exception as e:
            self.logger.warning(f"Could not load update config: {e}")

    async def _load_history(self):
        """Load update history"""
        try:
            history_file = Path(os.path.dirname(__file__)) / ".." / ".." / "data" / "update_history.json"

            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.update_history = data.get("history", [])

        except Exception as e:
            self.logger.warning(f"Could not load update history: {e}")

    async def _record_update(self, version: str, status: str):
        """Record update in history"""
        try:
            update_record = {
                "version": version,
                "timestamp": time.time(),
                "status": status,
                "previous_version": self.current_version
            }

            self.update_history.append(update_record)

            # Keep only recent history
            if len(self.update_history) > 50:
                self.update_history = self.update_history[-50:]

            # Save history
            history_file = Path(os.path.dirname(__file__)) / ".." / ".." / "data" / "update_history.json"
            history_file.parent.mkdir(exist_ok=True)

            with open(history_file, 'w') as f:
                json.dump({"history": self.update_history}, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error recording update: {e}")

    async def _notify_update_available(self, update_info: Dict[str, Any]):
        """Notify callbacks about available update"""
        for callback in self.update_callbacks:
            try:
                await callback("update_available", update_info)
            except Exception as e:
                self.logger.error(f"Error in update callback: {e}")

    async def _notify_update_installed(self, install_info: Dict[str, Any]):
        """Notify callbacks about installed update"""
        for callback in self.update_callbacks:
            try:
                await callback("update_installed", install_info)
            except Exception as e:
                self.logger.error(f"Error in install callback: {e}")

    def add_update_callback(self, callback: Callable):
        """Add update notification callback"""
        self.update_callbacks.append(callback)

    def remove_update_callback(self, callback: Callable):
        """Remove update notification callback"""
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)

    def get_update_status(self) -> Dict[str, Any]:
        """Get current update status"""
        return {
            "current_version": self.current_version,
            "latest_version": self.latest_version,
            "update_available": self.update_available,
            "installing": self.installing,
            "download_progress": self.download_progress,
            "auto_update_enabled": self.update_config["auto_update"],
            "last_update_history": self.update_history[-5:] if self.update_history else []
        }

    async def _cleanup_temp_files(self):
        """Clean up temporary update files"""
        try:
            temp_dir = Path(tempfile.gettempdir())

            # Remove old update files
            for item in temp_dir.iterdir():
                if item.name.startswith(("jarvis_update", "jarvis_backup")):
                    try:
                        if item.is_file():
                            item.unlink()
                        else:
                            shutil.rmtree(item)
                    except Exception as e:
                        self.logger.warning(f"Could not remove temp file {item}: {e}")

        except Exception as e:
            self.logger.error(f"Error cleaning temp files: {e}")

    async def shutdown(self):
        """Shutdown update manager"""
        try:
            self.logger.info("Shutting down update manager...")

            # Save configuration
            config_file = Path(os.path.dirname(__file__)) / ".." / ".." / "config" / "updater.json"
            config_file.parent.mkdir(exist_ok=True)

            with open(config_file, 'w') as f:
                json.dump(self.update_config, f, indent=2)

            self.logger.info("Update manager shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down update manager: {e}")