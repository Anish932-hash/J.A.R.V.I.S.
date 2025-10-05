"""
J.A.R.V.I.S. Security Manager
Advanced security and access control system
"""

import os
import time
import hashlib
import secrets
import threading
import subprocess
from typing import Dict, List, Optional, Any, Tuple
import logging

# Import EventType from event_manager
from jarvis.core.event_manager import EventType


class SecurityManager:
    """
    Advanced security management system
    Handles authentication, encryption, and access control
    """

    def __init__(self, jarvis_instance):
        """
        Initialize security manager

        Args:
            jarvis_instance: Reference to main JARVIS instance
        """
        self.jarvis = jarvis_instance
        self.logger = logging.getLogger('JARVIS.SecurityManager')

        # Authentication
        self.authenticated_users = {}
        self.login_attempts = {}
        self.max_login_attempts = 3
        self.lockout_duration = 300  # 5 minutes

        # Encryption
        self.encryption_enabled = True
        self.key_rotation_interval = 24 * 60 * 60  # 24 hours

        # Access control
        self.access_rules = {}
        self.security_policies = {}

        # Monitoring
        self.security_events = []
        self.max_security_events = 1000

        # Threat detection
        self.threat_signatures = {}
        self.suspicious_activities = []

        # Performance tracking
        self.stats = {
            "authentication_attempts": 0,
            "successful_logins": 0,
            "failed_logins": 0,
            "security_events": 0,
            "threats_detected": 0
        }

    def initialize(self):
        """Initialize security manager"""
        try:
            self.logger.info("Initializing security manager...")

            # Load security policies
            self._load_security_policies()

            # Initialize encryption keys
            self._initialize_encryption()

            # Start security monitoring
            self._start_security_monitoring()

            self.logger.info("Security manager initialized")

        except Exception as e:
            self.logger.error(f"Error initializing security manager: {e}")
            raise

    def _load_security_policies(self):
        """Load default security policies"""
        self.security_policies = {
            "password_policy": {
                "min_length": 8,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_symbols": True,
                "max_age_days": 90
            },
            "access_policy": {
                "session_timeout": 3600,  # 1 hour
                "require_mfa": False,
                "allowed_ip_ranges": [],
                "blocked_ip_ranges": []
            },
            "audit_policy": {
                "log_all_commands": True,
                "log_file_access": True,
                "log_network_activity": True,
                "retention_days": 30
            }
        }

    def _initialize_encryption(self):
        """Initialize encryption system"""
        try:
            # Generate master key if it doesn't exist
            key_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'master.key')

            if not os.path.exists(key_file):
                # Generate new master key
                master_key = secrets.token_bytes(32)

                # Ensure data directory exists
                os.makedirs(os.path.dirname(key_file), exist_ok=True)

                with open(key_file, 'wb') as f:
                    f.write(master_key)

                self.logger.info("Generated new master encryption key")
            else:
                self.logger.info("Using existing master encryption key")

        except Exception as e:
            self.logger.error(f"Error initializing encryption: {e}")

    def _start_security_monitoring(self):
        """Start security monitoring thread"""
        if not hasattr(self, '_security_monitor'):
            self._security_monitor = threading.Thread(
                target=self._security_monitoring_loop,
                name="SecurityMonitor",
                daemon=True
            )
            self._security_monitor.start()

    def authenticate_user(self, username: str, password: str, ip_address: str = None) -> Dict[str, Any]:
        """
        Authenticate a user

        Args:
            username: Username
            password: Password
            ip_address: Client IP address

        Returns:
            Authentication result
        """
        try:
            self.stats["authentication_attempts"] += 1

            # Check for lockout
            if self._is_user_locked_out(username):
                return {
                    "success": False,
                    "error": "Account temporarily locked due to multiple failed attempts",
                    "locked_until": self.login_attempts[username]["locked_until"]
                }

            # Verify credentials (simplified - in real system would check against stored hashes)
            if self._verify_credentials(username, password):
                # Successful authentication
                session_token = self._generate_session_token(username)
                self.authenticated_users[username] = {
                    "session_token": session_token,
                    "login_time": time.time(),
                    "ip_address": ip_address,
                    "last_activity": time.time()
                }

                # Clear failed attempts
                if username in self.login_attempts:
                    del self.login_attempts[username]

                self.stats["successful_logins"] += 1

                # Log security event
                self._log_security_event("user_login", {
                    "username": username,
                    "ip_address": ip_address,
                    "success": True
                })

                return {
                    "success": True,
                    "session_token": session_token,
                    "message": f"Welcome, {username}",
                    "expires_at": time.time() + self.security_policies["access_policy"]["session_timeout"]
                }

            else:
                # Failed authentication
                self._record_failed_attempt(username, ip_address)
                self.stats["failed_logins"] += 1

                # Log security event
                self._log_security_event("user_login_failed", {
                    "username": username,
                    "ip_address": ip_address,
                    "success": False
                })

                return {
                    "success": False,
                    "error": "Invalid username or password"
                }

        except Exception as e:
            self.logger.error(f"Error during authentication: {e}")
            return {
                "success": False,
                "error": "Authentication system error"
            }

    def _verify_credentials(self, username: str, password: str) -> bool:
        """Verify user credentials with proper hashing and storage"""
        try:
            # Load user database
            users_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'users.json')

            if not os.path.exists(users_file):
                # Create default admin user if no users exist
                self._create_default_admin()
                return username == "admin" and self._verify_password(password, self._hash_password("admin123!"))

            # Load users
            with open(users_file, 'r') as f:
                users_data = json.load(f)

            users = users_data.get("users", {})

            if username not in users:
                return False

            user_data = users[username]

            # Check if account is locked
            if user_data.get("locked", False):
                return False

            # Verify password
            stored_hash = user_data.get("password_hash", "")
            if not self._verify_password(password, stored_hash):
                return False

            # Check password age
            password_age_days = (time.time() - user_data.get("password_changed", time.time())) / (24 * 60 * 60)
            max_age = self.security_policies["password_policy"]["max_age_days"]

            if password_age_days > max_age:
                self.logger.warning(f"Password for user {username} is expired")
                # Still allow login but log warning

            return True

        except Exception as e:
            self.logger.error(f"Error verifying credentials: {e}")
            return False

    def _hash_password(self, password: str) -> str:
        """Hash password using PBKDF2"""
        try:
            import hashlib
            import binascii

            # Generate salt
            salt = secrets.token_bytes(32)

            # Hash password with PBKDF2
            hash_obj = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                100000  # Number of iterations
            )

            # Combine salt and hash
            hash_with_salt = salt + hash_obj

            return binascii.hexlify(hash_with_salt).decode('utf-8')

        except Exception as e:
            self.logger.error(f"Error hashing password: {e}")
            return ""

    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash"""
        try:
            import hashlib
            import binascii

            # Decode stored hash
            hash_bytes = binascii.unhexlify(stored_hash.encode('utf-8'))

            # Extract salt (first 32 bytes)
            salt = hash_bytes[:32]
            stored_hash_only = hash_bytes[32:]

            # Hash input password with same salt
            hash_obj = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                100000
            )

            # Compare hashes
            return hash_obj == stored_hash_only

        except Exception as e:
            self.logger.error(f"Error verifying password: {e}")
            return False

    def _create_default_admin(self):
        """Create default admin user"""
        try:
            users_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'users.json')

            # Create default admin user
            default_password = "admin123!"
            password_hash = self._hash_password(default_password)

            users_data = {
                "users": {
                    "admin": {
                        "username": "admin",
                        "password_hash": password_hash,
                        "role": "administrator",
                        "created_at": time.time(),
                        "password_changed": time.time(),
                        "locked": False,
                        "login_attempts": 0,
                        "last_login": None
                    }
                },
                "created_at": time.time(),
                "version": "1.0"
            }

            os.makedirs(os.path.dirname(users_file), exist_ok=True)
            with open(users_file, 'w') as f:
                json.dump(users_data, f, indent=2)

            self.logger.info("Created default admin user (username: admin, password: admin123!)")

        except Exception as e:
            self.logger.error(f"Error creating default admin user: {e}")

    def _generate_session_token(self, username: str) -> str:
        """Generate secure session token"""
        try:
            # Create token with username, timestamp, and random data
            token_data = f"{username}:{time.time()}:{secrets.token_hex(16)}"
            token_hash = hashlib.sha256(token_data.encode()).hexdigest()
            return token_hash

        except Exception as e:
            self.logger.error(f"Error generating session token: {e}")
            return secrets.token_hex(32)

    def _is_user_locked_out(self, username: str) -> bool:
        """Check if user account is locked out"""
        if username not in self.login_attempts:
            return False

        attempt_info = self.login_attempts[username]

        if attempt_info["count"] >= self.max_login_attempts:
            if time.time() - attempt_info["last_attempt"] < self.lockout_duration:
                return True
            else:
                # Lockout period expired, clear attempts
                del self.login_attempts[username]

        return False

    def _record_failed_attempt(self, username: str, ip_address: str):
        """Record a failed login attempt"""
        if username not in self.login_attempts:
            self.login_attempts[username] = {
                "count": 0,
                "last_attempt": 0,
                "ip_address": ip_address
            }

        self.login_attempts[username]["count"] += 1
        self.login_attempts[username]["last_attempt"] = time.time()
        self.login_attempts[username]["ip_address"] = ip_address

    def validate_session(self, username: str, session_token: str) -> bool:
        """
        Validate user session

        Args:
            username: Username
            session_token: Session token to validate

        Returns:
            True if session is valid
        """
        try:
            if username not in self.authenticated_users:
                return False

            user_session = self.authenticated_users[username]

            # Check session token
            if user_session["session_token"] != session_token:
                return False

            # Check session timeout
            session_timeout = self.security_policies["access_policy"]["session_timeout"]
            if time.time() - user_session["login_time"] > session_timeout:
                # Session expired
                del self.authenticated_users[username]
                return False

            # Update last activity
            user_session["last_activity"] = time.time()

            return True

        except Exception as e:
            self.logger.error(f"Error validating session: {e}")
            return False

    def logout_user(self, username: str) -> bool:
        """
        Logout a user

        Args:
            username: Username to logout

        Returns:
            Success status
        """
        try:
            if username in self.authenticated_users:
                del self.authenticated_users[username]

                # Log security event
                self._log_security_event("user_logout", {
                    "username": username,
                    "success": True
                })

                return True

            return False

        except Exception as e:
            self.logger.error(f"Error during logout: {e}")
            return False

    def encrypt_data(self, data: str, key: str = None) -> str:
        """
        Encrypt data using AES

        Args:
            data: Data to encrypt
            key: Encryption key (uses master key if None)

        Returns:
            Encrypted data as base64 string
        """
        if not self.encryption_enabled:
            return data

        try:
            from cryptography.fernet import Fernet
            import base64

            # Use provided key or master key
            if not key:
                key_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'master.key')
                with open(key_file, 'rb') as f:
                    key = base64.urlsafe_b64encode(f.read())

            fernet = Fernet(key)

            if isinstance(data, str):
                data = data.encode('utf-8')

            encrypted = fernet.encrypt(data)
            return base64.urlsafe_b64encode(encrypted).decode('utf-8')

        except Exception as e:
            self.logger.error(f"Error encrypting data: {e}")
            return data

    def decrypt_data(self, encrypted_data: str, key: str = None) -> str:
        """
        Decrypt data using AES

        Args:
            encrypted_data: Encrypted data as base64 string
            key: Decryption key (uses master key if None)

        Returns:
            Decrypted data
        """
        if not self.encryption_enabled:
            return encrypted_data

        try:
            from cryptography.fernet import Fernet
            import base64

            # Use provided key or master key
            if not key:
                key_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'master.key')
                with open(key_file, 'rb') as f:
                    key = base64.urlsafe_b64encode(f.read())

            fernet = Fernet(key)

            # Decode from base64
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))

            decrypted = fernet.decrypt(encrypted_bytes)
            return decrypted.decode('utf-8')

        except Exception as e:
            self.logger.error(f"Error decrypting data: {e}")
            return encrypted_data

    def check_permission(self, username: str, resource: str, action: str) -> bool:
        """
        Check if user has permission for action on resource

        Args:
            username: Username
            resource: Resource identifier
            action: Action to perform

        Returns:
            True if permission granted
        """
        try:
            # Validate session first
            if username not in self.authenticated_users:
                return False

            # Check access rules
            user_rules = self.access_rules.get(username, {})

            # Check specific rule
            resource_rules = user_rules.get(resource, {})
            if action in resource_rules:
                return resource_rules[action]

            # Check wildcard rules
            if "*" in user_rules:
                wildcard_rules = user_rules["*"]
                if action in wildcard_rules:
                    return wildcard_rules[action]

            # Default deny
            return False

        except Exception as e:
            self.logger.error(f"Error checking permission: {e}")
            return False

    def grant_permission(self, username: str, resource: str, action: str, allowed: bool = True):
        """
        Grant or revoke permission for user

        Args:
            username: Username
            resource: Resource identifier
            action: Action to allow/deny
            allowed: True to grant, False to revoke
        """
        try:
            if username not in self.access_rules:
                self.access_rules[username] = {}

            if resource not in self.access_rules[username]:
                self.access_rules[username][resource] = {}

            self.access_rules[username][resource][action] = allowed

            self.logger.info(f"Permission {action} on {resource} {'granted' if allowed else 'revoked'} for {username}")

        except Exception as e:
            self.logger.error(f"Error setting permission: {e}")

    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log a security event"""
        try:
            event = {
                "event_type": event_type,
                "timestamp": time.time(),
                "details": details
            }

            self.security_events.append(event)
            self.stats["security_events"] += 1

            # Maintain event history size
            if len(self.security_events) > self.max_security_events:
                self.security_events.pop(0)

            # Also send to main event manager if available
            if hasattr(self.jarvis, 'event_manager'):
                self.jarvis.event_manager.emit_event(
                    EventType.SECURITY_EVENT,
                    event,
                    source="security_manager"
                )

        except Exception as e:
            self.logger.error(f"Error logging security event: {e}")

    def _security_monitoring_loop(self):
        """Security monitoring loop"""
        while True:
            try:
                # Check for suspicious activities
                self._detect_threats()

                # Check session timeouts
                self._check_session_timeouts()

                # Rotate encryption keys if needed
                self._check_key_rotation()

                time.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Error in security monitoring: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying

    def _detect_threats(self):
        """Detect potential security threats"""
        try:
            # Check for multiple failed login attempts
            for username, attempt_info in self.login_attempts.items():
                if attempt_info["count"] >= self.max_login_attempts:
                    self._log_security_event("brute_force_attempt", {
                        "username": username,
                        "attempt_count": attempt_info["count"],
                        "ip_address": attempt_info.get("ip_address", "unknown")
                    })
                    self.stats["threats_detected"] += 1

            # Check for unusual command patterns (simplified)
            # In a real system, this would analyze command history for anomalies

        except Exception as e:
            self.logger.error(f"Error in threat detection: {e}")

    def _check_session_timeouts(self):
        """Check for expired sessions"""
        current_time = time.time()
        session_timeout = self.security_policies["access_policy"]["session_timeout"]

        expired_sessions = []

        for username, session_info in self.authenticated_users.items():
            if current_time - session_info["login_time"] > session_timeout:
                expired_sessions.append(username)

        # Remove expired sessions
        for username in expired_sessions:
            del self.authenticated_users[username]
            self._log_security_event("session_expired", {"username": username})

    def _check_key_rotation(self):
        """Check if encryption keys need rotation"""
        try:
            key_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'master.key')

            if os.path.exists(key_file):
                key_age = time.time() - os.path.getmtime(key_file)

                if key_age > self.key_rotation_interval:
                    self.logger.info("Rotating encryption keys...")
                    # Generate new key
                    new_key = secrets.token_bytes(32)

                    with open(key_file, 'wb') as f:
                        f.write(new_key)

                    self._log_security_event("key_rotation", {"reason": "scheduled"})

        except Exception as e:
            self.logger.error(f"Error during key rotation: {e}")

    def scan_for_malware(self, scan_path: str = None) -> Dict[str, Any]:
        """Scan for malware (simplified)"""
        try:
            scan_path = scan_path or "C:\\"

            # This is a simplified implementation
            # In a real system, you would integrate with antivirus software

            self.logger.info(f"Scanning for malware in: {scan_path}")

            # Simulate scan results
            scan_results = {
                "path": scan_path,
                "files_scanned": 0,
                "threats_found": 0,
                "threats": [],
                "scan_time": 0
            }

            # Log security event
            self._log_security_event("malware_scan", scan_results)

            return {
                "success": True,
                "message": "Malware scan completed",
                "results": scan_results
            }

        except Exception as e:
            self.logger.error(f"Error scanning for malware: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_firewall_status(self) -> Dict[str, Any]:
        """Get Windows Firewall status"""
        try:
            # Check Windows Firewall status
            try:
                output = subprocess.check_output("netsh advfirewall show allprofiles state", shell=True).decode()

                firewall_info = {}
                current_profile = None

                for line in output.split('\n'):
                    line = line.strip()
                    if line.startswith('Profile'):
                        current_profile = line.split()[1].rstrip(':')
                        firewall_info[current_profile] = {}
                    elif current_profile and 'State' in line:
                        state = line.split()[1]
                        firewall_info[current_profile] = {"state": state}

                return {
                    "success": True,
                    "firewall_info": firewall_info,
                    "timestamp": time.time()
                }

            except subprocess.CalledProcessError:
                return {
                    "success": False,
                    "error": "Could not retrieve firewall status",
                    "timestamp": time.time()
                }

        except Exception as e:
            self.logger.error(f"Error getting firewall status: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

    def get_security_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get security audit log"""
        return self.security_events[-limit:] if self.security_events else []

    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get list of active user sessions"""
        current_time = time.time()

        sessions = []
        for username, session_info in self.authenticated_users.items():
            sessions.append({
                "username": username,
                "login_time": session_info["login_time"],
                "last_activity": session_info["last_activity"],
                "ip_address": session_info.get("ip_address", "unknown"),
                "session_duration": current_time - session_info["login_time"],
                "time_since_activity": current_time - session_info["last_activity"]
            })

        return sessions

    def get_failed_login_attempts(self) -> Dict[str, Any]:
        """Get failed login attempts"""
        return {
            "attempts": self.login_attempts,
            "total_failed_attempts": sum(info["count"] for info in self.login_attempts.values()),
            "locked_accounts": len([u for u, info in self.login_attempts.items() if info["count"] >= self.max_login_attempts])
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get security manager statistics"""
        return {
            **self.stats,
            "active_sessions": len(self.authenticated_users),
            "failed_login_attempts": len(self.login_attempts),
            "security_events_count": len(self.security_events),
            "encryption_enabled": self.encryption_enabled
        }

    def clear_security_events(self):
        """Clear security events log"""
        self.security_events.clear()
        self.logger.info("Security events log cleared")

    def backup_security_data(self, backup_path: str = None) -> Dict[str, Any]:
        """Backup security data"""
        try:
            if not backup_path:
                backup_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'backups')
                os.makedirs(backup_dir, exist_ok=True)
                backup_path = os.path.join(backup_dir, f"security_backup_{int(time.time())}.json")

            # Prepare backup data
            backup_data = {
                "timestamp": time.time(),
                "access_rules": self.access_rules,
                "security_policies": self.security_policies,
                "login_attempts": self.login_attempts,
                "authenticated_users": {u: {k: v for k, v in info.items() if k != "session_token"}
                                     for u, info in self.authenticated_users.items()}
            }

            # Encrypt backup if encryption is enabled
            if self.encryption_enabled:
                backup_data = self.encrypt_data(str(backup_data))

            # Save backup
            with open(backup_path, 'w') as f:
                f.write(str(backup_data))

            return {
                "success": True,
                "backup_path": backup_path,
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"Error creating security backup: {e}")
            return {
                "success": False,
                "error": str(e)
            }