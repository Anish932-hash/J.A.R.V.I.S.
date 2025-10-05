"""
J.A.R.V.I.S. File Manager
Advanced file system automation and management
"""

import os
import shutil
import time
import hashlib
import zipfile
import rarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging


class FileManager:
    """
    Advanced file management system
    Handles file operations, search, organization, and automation
    """

    def __init__(self, jarvis_instance):
        """
        Initialize file manager

        Args:
            jarvis_instance: Reference to main JARVIS instance
        """
        self.jarvis = jarvis_instance
        self.logger = logging.getLogger('JARVIS.FileManager')

        # File operation tracking
        self.operation_history = []
        self.max_history_size = 1000

        # File type categories
        self.file_categories = {
            "documents": [".doc", ".docx", ".pdf", ".txt", ".rtf", ".odt"],
            "spreadsheets": [".xls", ".xlsx", ".csv", ".ods"],
            "presentations": [".ppt", ".pptx", ".odp"],
            "images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"],
            "videos": [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"],
            "audio": [".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma"],
            "archives": [".zip", ".rar", ".7z", ".tar", ".gz"],
            "code": [".py", ".js", ".html", ".css", ".java", ".cpp", ".c", ".php"],
            "executables": [".exe", ".msi", ".bat", ".cmd", ".com"]
        }

        # Search settings
        self.search_settings = {
            "case_sensitive": False,
            "include_hidden": False,
            "max_results": 1000,
            "search_timeout": 30
        }

        # Performance tracking
        self.stats = {
            "files_created": 0,
            "files_deleted": 0,
            "files_moved": 0,
            "files_copied": 0,
            "searches_performed": 0,
            "bytes_processed": 0
        }

    def initialize(self):
        """Initialize file manager"""
        try:
            self.logger.info("Initializing file manager...")

            # Create necessary directories
            self._ensure_directories()

            self.logger.info("File manager initialized")

        except Exception as e:
            self.logger.error(f"Error initializing file manager: {e}")
            raise

    def _ensure_directories(self):
        """Ensure necessary directories exist"""
        directories = [
            os.path.join(os.path.dirname(__file__), '..', 'data', 'temp'),
            os.path.join(os.path.dirname(__file__), '..', 'data', 'backups'),
            os.path.join(os.path.dirname(__file__), '..', 'data', 'downloads')
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def create_file(self,
                    file_path: str,
                    content: str = "",
                    encoding: str = "utf-8") -> Dict[str, Any]:
        """
        Create a new file

        Args:
            file_path: Path where to create the file
            content: Initial content for the file
            encoding: File encoding

        Returns:
            Operation result
        """
        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Create file
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)

            # Get file info
            file_size = os.path.getsize(file_path)

            # Update tracking
            self._add_to_history("create", file_path, {"size": file_size})
            self.stats["files_created"] += 1
            self.stats["bytes_processed"] += file_size

            self.logger.info(f"Created file: {file_path} ({file_size} bytes)")

            return {
                "success": True,
                "message": f"File created successfully: {file_path}",
                "file_path": file_path,
                "file_size": file_size,
                "encoding": encoding
            }

        except Exception as e:
            self.logger.error(f"Error creating file {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }

    def write_file(self,
                   file_path: str,
                   content: str,
                   encoding: str = "utf-8") -> Dict[str, Any]:
        """
        Write content to a file (convenience method)

        Args:
            file_path: Path of the file to write
            content: Content to write
            encoding: File encoding

        Returns:
            Operation result
        """
        return self.create_file(file_path, content, encoding)

    def read_file(self, file_path: str, encoding: str = "utf-8") -> str:
        """
        Read content from a file

        Args:
            file_path: Path of the file to read
            encoding: File encoding

        Returns:
            File content as string
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()

            # Update tracking
            file_size = os.path.getsize(file_path)
            self.stats["bytes_processed"] += file_size

            return content

        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            raise

    def delete_file(self, file_path: str, trash: bool = True) -> Dict[str, Any]:
        """
        Delete a file

        Args:
            file_path: Path of file to delete
            trash: Move to recycle bin instead of permanent delete

        Returns:
            Operation result
        """
        try:
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "error": f"File not found: {file_path}"
                }

            # Get file info before deletion
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)

            if trash:
                # Move to recycle bin (Windows)
                try:
                    import winshell
                    winshell.delete_file(file_path, no_confirm=True)
                    delete_type = "recycle_bin"
                except ImportError:
                    # Fallback to permanent delete
                    os.remove(file_path)
                    delete_type = "permanent"
            else:
                # Permanent delete
                os.remove(file_path)
                delete_type = "permanent"

            # Update tracking
            self._add_to_history("delete", file_path, {"size": file_size, "type": delete_type})
            self.stats["files_deleted"] += 1
            self.stats["bytes_processed"] += file_size

            self.logger.info(f"Deleted file: {file_path} ({delete_type})")

            return {
                "success": True,
                "message": f"File deleted successfully: {file_path}",
                "file_path": file_path,
                "file_size": file_size,
                "delete_type": delete_type
            }

        except Exception as e:
            self.logger.error(f"Error deleting file {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }

    def copy_file(self,
                 source_path: str,
                 destination_path: str,
                 overwrite: bool = False) -> Dict[str, Any]:
        """
        Copy a file

        Args:
            source_path: Source file path
            destination_path: Destination file path
            overwrite: Overwrite if destination exists

        Returns:
            Operation result
        """
        try:
            if not os.path.exists(source_path):
                return {
                    "success": False,
                    "error": f"Source file not found: {source_path}"
                }

            # Check if destination exists
            if os.path.exists(destination_path) and not overwrite:
                return {
                    "success": False,
                    "error": f"Destination file already exists: {destination_path}"
                }

            # Ensure destination directory exists
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)

            # Copy file
            shutil.copy2(source_path, destination_path)

            # Get file sizes
            source_size = os.path.getsize(source_path)
            dest_size = os.path.getsize(destination_path)

            # Update tracking
            self._add_to_history("copy", source_path, {
                "destination": destination_path,
                "size": source_size
            })
            self.stats["files_copied"] += 1
            self.stats["bytes_processed"] += source_size

            self.logger.info(f"Copied file: {source_path} -> {destination_path}")

            return {
                "success": True,
                "message": f"File copied successfully: {source_path} -> {destination_path}",
                "source_path": source_path,
                "destination_path": destination_path,
                "source_size": source_size,
                "destination_size": dest_size
            }

        except Exception as e:
            self.logger.error(f"Error copying file {source_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "source_path": source_path,
                "destination_path": destination_path
            }

    def move_file(self,
                 source_path: str,
                 destination_path: str,
                 overwrite: bool = False) -> Dict[str, Any]:
        """
        Move a file

        Args:
            source_path: Source file path
            destination_path: Destination file path
            overwrite: Overwrite if destination exists

        Returns:
            Operation result
        """
        try:
            if not os.path.exists(source_path):
                return {
                    "success": False,
                    "error": f"Source file not found: {source_path}"
                }

            # Check if destination exists
            if os.path.exists(destination_path) and not overwrite:
                return {
                    "success": False,
                    "error": f"Destination file already exists: {destination_path}"
                }

            # Ensure destination directory exists
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)

            # Move file
            shutil.move(source_path, destination_path)

            # Get file size
            file_size = os.path.getsize(destination_path)

            # Update tracking
            self._add_to_history("move", source_path, {
                "destination": destination_path,
                "size": file_size
            })
            self.stats["files_moved"] += 1
            self.stats["bytes_processed"] += file_size

            self.logger.info(f"Moved file: {source_path} -> {destination_path}")

            return {
                "success": True,
                "message": f"File moved successfully: {source_path} -> {destination_path}",
                "source_path": source_path,
                "destination_path": destination_path,
                "file_size": file_size
            }

        except Exception as e:
            self.logger.error(f"Error moving file {source_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "source_path": source_path,
                "destination_path": destination_path
            }

    def search_files(self,
                    query: str,
                    search_path: str = None,
                    file_types: List[str] = None,
                    recursive: bool = True) -> Dict[str, Any]:
        """
        Search for files

        Args:
            query: Search query (filename or content)
            search_path: Directory to search in (current dir if None)
            file_types: List of file extensions to include
            recursive: Search subdirectories

        Returns:
            Search results
        """
        try:
            search_path = search_path or os.getcwd()
            results = []

            if not os.path.exists(search_path):
                return {
                    "success": False,
                    "error": f"Search path not found: {search_path}"
                }

            # Update stats
            self.stats["searches_performed"] += 1

            # Search for files
            for root, dirs, files in os.walk(search_path):
                if not recursive and root != search_path:
                    break

                for file in files:
                    file_path = os.path.join(root, file)
                    file_name = os.path.basename(file_path)
                    file_ext = os.path.splitext(file_path)[1].lower()

                    # Check file type filter
                    if file_types and file_ext not in file_types:
                        continue

                    # Check if query matches filename
                    if self._matches_query(query, file_name):
                        try:
                            file_info = self._get_file_info(file_path)
                            results.append(file_info)

                            if len(results) >= self.search_settings["max_results"]:
                                break

                        except Exception as e:
                            self.logger.debug(f"Error getting info for {file_path}: {e}")

                if len(results) >= self.search_settings["max_results"]:
                    break

            self.logger.info(f"File search completed: {len(results)} results for '{query}'")

            return {
                "success": True,
                "query": query,
                "search_path": search_path,
                "results_count": len(results),
                "results": results[:self.search_settings["max_results"]],
                "truncated": len(results) > self.search_settings["max_results"]
            }

        except Exception as e:
            self.logger.error(f"Error searching files: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }

    def _matches_query(self, query: str, filename: str) -> bool:
        """Check if query matches filename"""
        if not self.search_settings["case_sensitive"]:
            query = query.lower()
            filename = filename.lower()

        return query in filename

    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get comprehensive file information"""
        try:
            stat = os.stat(file_path)

            return {
                "path": file_path,
                "name": os.path.basename(file_path),
                "extension": os.path.splitext(file_path)[1],
                "size_bytes": stat.st_size,
                "size_human": self._format_file_size(stat.st_size),
                "modified_time": stat.st_mtime,
                "created_time": stat.st_ctime,
                "is_hidden": self._is_hidden_file(file_path),
                "category": self._get_file_category(file_path),
                "checksum": self._get_file_checksum(file_path)
            }

        except Exception as e:
            self.logger.debug(f"Error getting file info for {file_path}: {e}")
            return {"path": file_path, "error": str(e)}

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    def _is_hidden_file(self, file_path: str) -> bool:
        """Check if file is hidden"""
        try:
            return os.path.basename(file_path).startswith('.')
        except:
            return False

    def _get_file_category(self, file_path: str) -> str:
        """Get file category based on extension"""
        extension = os.path.splitext(file_path)[1].lower()

        for category, extensions in self.file_categories.items():
            if extension in extensions:
                return category

        return "other"

    def _get_file_checksum(self, file_path: str, algorithm: str = "md5") -> str:
        """Get file checksum"""
        try:
            hash_func = hashlib.md5() if algorithm == "md5" else hashlib.sha256()

            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)

            return hash_func.hexdigest()

        except Exception as e:
            self.logger.debug(f"Error getting checksum for {file_path}: {e}")
            return ""

    def _add_to_history(self, operation: str, file_path: str, metadata: Dict[str, Any]):
        """Add operation to history"""
        self.operation_history.append({
            "operation": operation,
            "file_path": file_path,
            "timestamp": time.time(),
            "metadata": metadata
        })

        # Maintain history size
        if len(self.operation_history) > self.max_history_size:
            self.operation_history.pop(0)

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get detailed file information"""
        if not os.path.exists(file_path):
            return {"error": "File not found"}

        return self._get_file_info(file_path)

    def list_directory(self,
                      directory_path: str,
                      recursive: bool = False,
                      show_hidden: bool = False) -> Dict[str, Any]:
        """
        List directory contents

        Args:
            directory_path: Directory to list
            recursive: List subdirectories
            show_hidden: Include hidden files

        Returns:
            Directory listing
        """
        try:
            if not os.path.exists(directory_path):
                return {
                    "success": False,
                    "error": f"Directory not found: {directory_path}"
                }

            contents = []

            if recursive:
                for root, dirs, files in os.walk(directory_path):
                    for file in files:
                        file_path = os.path.join(root, file)

                        if not show_hidden and self._is_hidden_file(file_path):
                            continue

                        contents.append(self._get_file_info(file_path))
            else:
                for item in os.listdir(directory_path):
                    item_path = os.path.join(directory_path, item)

                    if not show_hidden and self._is_hidden_file(item_path):
                        continue

                    if os.path.isfile(item_path):
                        contents.append(self._get_file_info(item_path))
                    else:
                        # Directory info
                        contents.append({
                            "path": item_path,
                            "name": item,
                            "type": "directory",
                            "size_bytes": 0
                        })

            return {
                "success": True,
                "directory": directory_path,
                "contents": contents,
                "total_files": len(contents)
            }

        except Exception as e:
            self.logger.error(f"Error listing directory {directory_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "directory": directory_path
            }

    def create_directory(self, directory_path: str) -> Dict[str, Any]:
        """Create a directory"""
        try:
            os.makedirs(directory_path, exist_ok=True)

            self._add_to_history("create_directory", directory_path, {})

            return {
                "success": True,
                "message": f"Directory created: {directory_path}",
                "directory_path": directory_path
            }

        except Exception as e:
            self.logger.error(f"Error creating directory {directory_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "directory_path": directory_path
            }

    def delete_directory(self, directory_path: str, recursive: bool = False) -> Dict[str, Any]:
        """Delete a directory"""
        try:
            if not os.path.exists(directory_path):
                return {
                    "success": False,
                    "error": f"Directory not found: {directory_path}"
                }

            if recursive:
                shutil.rmtree(directory_path)
                delete_type = "recursive"
            else:
                os.rmdir(directory_path)
                delete_type = "single"

            self._add_to_history("delete_directory", directory_path, {"type": delete_type})

            return {
                "success": True,
                "message": f"Directory deleted: {directory_path}",
                "directory_path": directory_path,
                "delete_type": delete_type
            }

        except Exception as e:
            self.logger.error(f"Error deleting directory {directory_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "directory_path": directory_path
            }

    def compress_files(self,
                      files: List[str],
                      archive_path: str,
                      format: str = "zip") -> Dict[str, Any]:
        """
        Compress files to archive

        Args:
            files: List of file paths to compress
            archive_path: Output archive path
            format: Archive format (zip, tar, etc.)

        Returns:
            Compression result
        """
        try:
            # Ensure archive directory exists
            os.makedirs(os.path.dirname(archive_path), exist_ok=True)

            if format.lower() == "zip":
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
                    for file_path in files:
                        if os.path.exists(file_path):
                            archive.write(file_path, os.path.basename(file_path))

            elif format.lower() == "tar":
                import tarfile
                with tarfile.open(archive_path, 'w') as archive:
                    for file_path in files:
                        if os.path.exists(file_path):
                            archive.add(file_path, os.path.basename(file_path))

            # Get archive size
            archive_size = os.path.getsize(archive_path)

            self._add_to_history("compress", ",".join(files), {
                "archive_path": archive_path,
                "format": format,
                "archive_size": archive_size
            })

            return {
                "success": True,
                "message": f"Files compressed to: {archive_path}",
                "archive_path": archive_path,
                "archive_size": archive_size,
                "format": format,
                "files_count": len(files)
            }

        except Exception as e:
            self.logger.error(f"Error compressing files: {e}")
            return {
                "success": False,
                "error": str(e),
                "archive_path": archive_path
            }

    def extract_archive(self,
                       archive_path: str,
                       extract_to: str = None) -> Dict[str, Any]:
        """
        Extract archive

        Args:
            archive_path: Archive file path
            extract_to: Directory to extract to (same as archive if None)

        Returns:
            Extraction result
        """
        try:
            if not os.path.exists(archive_path):
                return {
                    "success": False,
                    "error": f"Archive not found: {archive_path}"
                }

            extract_to = extract_to or os.path.splitext(archive_path)[0]

            # Ensure extraction directory exists
            os.makedirs(extract_to, exist_ok=True)

            # Extract based on format
            if archive_path.lower().endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as archive:
                    archive.extractall(extract_to)

            elif archive_path.lower().endswith('.rar'):
                with rarfile.RarFile(archive_path, 'r') as archive:
                    archive.extractall(extract_to)

            # Get extracted files
            extracted_files = []
            for root, dirs, files in os.walk(extract_to):
                for file in files:
                    extracted_files.append(os.path.join(root, file))

            self._add_to_history("extract", archive_path, {
                "extract_to": extract_to,
                "extracted_files": len(extracted_files)
            })

            return {
                "success": True,
                "message": f"Archive extracted to: {extract_to}",
                "archive_path": archive_path,
                "extract_to": extract_to,
                "extracted_files": len(extracted_files)
            }

        except Exception as e:
            self.logger.error(f"Error extracting archive {archive_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "archive_path": archive_path
            }

    def find_duplicates(self, directory_path: str) -> Dict[str, Any]:
        """Find duplicate files in directory"""
        try:
            checksums = {}
            duplicates = []

            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)

                    try:
                        checksum = self._get_file_checksum(file_path)
                        file_size = os.path.getsize(file_path)

                        if checksum in checksums:
                            # Check if it's actually a duplicate (same size and checksum)
                            existing = checksums[checksum]
                            if existing["size"] == file_size:
                                duplicates.append({
                                    "checksum": checksum,
                                    "files": existing["files"] + [file_path],
                                    "size": file_size
                                })
                        else:
                            checksums[checksum] = {
                                "files": [file_path],
                                "size": file_size
                            }

                    except Exception as e:
                        self.logger.debug(f"Error processing file {file_path}: {e}")

            return {
                "success": True,
                "directory": directory_path,
                "duplicates_found": len(duplicates),
                "duplicates": duplicates
            }

        except Exception as e:
            self.logger.error(f"Error finding duplicates in {directory_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "directory": directory_path
            }

    def organize_files(self,
                      source_directory: str,
                      target_directory: str = None,
                      organize_by: str = "type") -> Dict[str, Any]:
        """
        Organize files by type, date, or size

        Args:
            source_directory: Source directory to organize
            target_directory: Target directory (creates organized structure if None)
            organize_by: Organization method (type, date, size)

        Returns:
            Organization result
        """
        try:
            if not os.path.exists(source_directory):
                return {
                    "success": False,
                    "error": f"Source directory not found: {source_directory}"
                }

            target_directory = target_directory or source_directory
            organized_files = {}

            for file in os.listdir(source_directory):
                file_path = os.path.join(source_directory, file)

                if os.path.isfile(file_path):
                    if organize_by == "type":
                        category = self._get_file_category(file_path)
                        organize_path = os.path.join(target_directory, category)
                    elif organize_by == "date":
                        # Organize by modification date
                        mod_time = time.strftime("%Y-%m", time.localtime(os.path.getmtime(file_path)))
                        organize_path = os.path.join(target_directory, mod_time)
                    elif organize_by == "size":
                        # Organize by file size
                        size = os.path.getsize(file_path)
                        if size < 1024 * 1024:  # < 1MB
                            size_category = "small"
                        elif size < 100 * 1024 * 1024:  # < 100MB
                            size_category = "medium"
                        else:
                            size_category = "large"
                        organize_path = os.path.join(target_directory, size_category)
                    else:
                        organize_path = target_directory

                    # Create category directory
                    os.makedirs(organize_path, exist_ok=True)

                    # Move file
                    target_file = os.path.join(organize_path, file)
                    shutil.move(file_path, target_file)

                    if organize_path not in organized_files:
                        organized_files[organize_path] = []
                    organized_files[organize_path].append(target_file)

            return {
                "success": True,
                "message": f"Organized {sum(len(files) for files in organized_files.values())} files",
                "source_directory": source_directory,
                "target_directory": target_directory,
                "organized_by": organize_by,
                "organized_files": organized_files
            }

        except Exception as e:
            self.logger.error(f"Error organizing files in {source_directory}: {e}")
            return {
                "success": False,
                "error": str(e),
                "source_directory": source_directory
            }

    def get_disk_usage(self, path: str = None) -> Dict[str, Any]:
        """Get disk usage information"""
        try:
            path = path or "/"
            usage = shutil.disk_usage(path)

            return {
                "path": path,
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
                "total_human": self._format_file_size(usage.total),
                "used_human": self._format_file_size(usage.used),
                "free_human": self._format_file_size(usage.free),
                "usage_percent": (usage.used / usage.total) * 100
            }

        except Exception as e:
            self.logger.error(f"Error getting disk usage for {path}: {e}")
            return {
                "error": str(e),
                "path": path
            }

    def cleanup_temp_files(self, directory: str = None, older_than_days: int = 7) -> Dict[str, Any]:
        """Clean up temporary files"""
        try:
            directory = directory or os.path.join(os.path.dirname(__file__), '..', 'data', 'temp')
            cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)

            cleaned_files = []
            total_size = 0

            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)

                    try:
                        # Check if file is older than cutoff
                        if os.path.getmtime(file_path) < cutoff_time:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)

                            cleaned_files.append(file_path)
                            total_size += file_size

                    except Exception as e:
                        self.logger.debug(f"Error cleaning file {file_path}: {e}")

            return {
                "success": True,
                "message": f"Cleaned {len(cleaned_files)} temporary files",
                "directory": directory,
                "cleaned_files": len(cleaned_files),
                "total_size_bytes": total_size,
                "total_size_human": self._format_file_size(total_size)
            }

        except Exception as e:
            self.logger.error(f"Error cleaning temp files: {e}")
            return {
                "success": False,
                "error": str(e),
                "directory": directory
            }

    def get_operation_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get file operation history"""
        if limit:
            return self.operation_history[-limit:]
        return self.operation_history

    def get_stats(self) -> Dict[str, Any]:
        """Get file manager statistics"""
        return {
            **self.stats,
            "history_size": len(self.operation_history),
            "file_categories": len(self.file_categories),
            "search_settings": self.search_settings
        }

    def clear_history(self):
        """Clear operation history"""
        self.operation_history.clear()
        self.logger.info("File operation history cleared")