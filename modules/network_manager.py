"""
J.A.R.V.I.S. Network Manager
Advanced network monitoring and control system
"""

import os
import time
import socket
import threading
import subprocess
import psutil
import requests
from typing import Dict, List, Optional, Any, Tuple
import logging


class NetworkManager:
    """
    Advanced network management system
    Handles network monitoring, connectivity, and control
    """

    def __init__(self, jarvis_instance):
        """
        Initialize network manager

        Args:
            jarvis_instance: Reference to main JARVIS instance
        """
        self.jarvis = jarvis_instance
        self.logger = logging.getLogger('JARVIS.NetworkManager')

        # Network monitoring
        self.monitoring = False
        self.monitor_thread = None
        self.monitoring_interval = 5  # seconds

        # Network interfaces
        self.interfaces = {}
        self.default_interface = None

        # Connection tracking
        self.active_connections = []
        self.connection_history = []

        # Network statistics
        self.stats = {
            "data_sent": 0,
            "data_received": 0,
            "connections_made": 0,
            "connections_failed": 0,
            "monitoring_duration": 0
        }

        # Network configuration
        self.dns_servers = []
        self.gateway = None
        self.subnet_mask = None

    def initialize(self):
        """Initialize network manager"""
        try:
            self.logger.info("Initializing network manager...")

            # Get network interfaces
            self._scan_network_interfaces()

            # Set default interface
            self._set_default_interface()

            # Start monitoring
            self.start_monitoring()

            self.logger.info("Network manager initialized")

        except Exception as e:
            self.logger.error(f"Error initializing network manager: {e}")
            raise

    def _scan_network_interfaces(self):
        """Scan and catalog network interfaces"""
        try:
            self.interfaces = {}

            for interface_name, addresses in psutil.net_if_addrs().items():
                interface_info = {
                    "name": interface_name,
                    "addresses": [],
                    "status": "unknown"
                }

                for addr in addresses:
                    if addr.family.name == 'AF_INET':
                        interface_info["addresses"].append({
                            "address": addr.address,
                            "netmask": addr.netmask,
                            "broadcast": addr.broadcast
                        })
                    elif addr.family.name == 'AF_PACKET':
                        interface_info["mac_address"] = addr.address

                # Get interface status
                try:
                    stats = psutil.net_if_stats().get(interface_name, {})
                    interface_info["status"] = "up" if stats.get("isup", False) else "down"
                    interface_info["speed"] = stats.get("speed", 0)
                    interface_info["mtu"] = stats.get("mtu", 0)
                except:
                    pass

                self.interfaces[interface_name] = interface_info

            self.logger.info(f"Found {len(self.interfaces)} network interfaces")

        except Exception as e:
            self.logger.error(f"Error scanning interfaces: {e}")

    def _set_default_interface(self):
        """Set default network interface"""
        try:
            # Get default gateway
            gateways = psutil.net_if_addrs()
            route_info = None

            # Try to get routing information (Windows specific)
            try:
                route_output = subprocess.check_output("route print", shell=True).decode()
                for line in route_output.split('\n'):
                    if '0.0.0.0' in line and '0.0.0.0' in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            self.gateway = parts[2]
                            self.default_interface = parts[3]
                            break
            except:
                pass

            if not self.default_interface and self.interfaces:
                # Fallback to first available interface
                for name, info in self.interfaces.items():
                    if info["status"] == "up" and info["addresses"]:
                        self.default_interface = name
                        break

            self.logger.info(f"Default interface set to: {self.default_interface}")

        except Exception as e:
            self.logger.error(f"Error setting default interface: {e}")

    def start_monitoring(self):
        """Start network monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="NetworkMonitor",
                daemon=True
            )
            self.monitor_thread.start()
            self.logger.info("Network monitoring started")

    def stop_monitoring(self):
        """Stop network monitoring"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        self.logger.info("Network monitoring stopped")

    def _monitor_loop(self):
        """Main network monitoring loop"""
        start_time = time.time()

        self.logger.info("Network monitoring loop started")

        while self.monitoring:
            try:
                # Get network statistics
                network_stats = self.get_network_statistics()

                # Update connection tracking
                self._update_connections()

                # Check connectivity
                self._check_connectivity()

                # Update monitoring duration
                self.stats["monitoring_duration"] = time.time() - start_time

                time.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"Error in network monitoring: {e}")
                time.sleep(self.monitoring_interval * 2)

        self.logger.info("Network monitoring loop ended")

    def get_network_statistics(self) -> Dict[str, Any]:
        """Get current network statistics"""
        try:
            # Get network I/O counters
            io_counters = psutil.net_io_counters()

            # Get per-interface statistics
            interface_stats = {}
            for name, info in self.interfaces.items():
                try:
                    stats = psutil.net_io_counters(pernic=True).get(name, {})
                    if stats:
                        interface_stats[name] = {
                            "bytes_sent": stats.bytes_sent,
                            "bytes_recv": stats.bytes_recv,
                            "packets_sent": stats.packets_sent,
                            "packets_recv": stats.packets_recv,
                            "errin": stats.errin,
                            "errout": stats.errout,
                            "dropin": stats.dropin,
                            "dropout": stats.dropout
                        }
                except:
                    pass

            return {
                "total_bytes_sent": io_counters.bytes_sent,
                "total_bytes_recv": io_counters.bytes_recv,
                "total_packets_sent": io_counters.packets_sent,
                "total_packets_recv": io_counters.packets_recv,
                "interface_stats": interface_stats,
                "active_connections": len(self.active_connections),
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"Error getting network statistics: {e}")
            return {}

    def _update_connections(self):
        """Update active connections tracking"""
        try:
            connections = psutil.net_connections()

            # Update stats
            self.stats["data_sent"] = sum(conn.bytes_sent for conn in connections if hasattr(conn, 'bytes_sent') and conn.bytes_sent)
            self.stats["data_received"] = sum(conn.bytes_recv for conn in connections if hasattr(conn, 'bytes_recv') and conn.bytes_recv)

            self.active_connections = [
                {
                    "fd": conn.fd,
                    "family": conn.family.name,
                    "type": conn.type.name,
                    "local_addr": f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else None,
                    "remote_addr": f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else None,
                    "status": conn.status,
                    "pid": conn.pid
                }
                for conn in connections
                if conn.status != 'NONE'
            ]

        except Exception as e:
            self.logger.error(f"Error updating connections: {e}")

    def _check_connectivity(self):
        """Check network connectivity"""
        try:
            # Test basic connectivity
            test_sites = [
                "8.8.8.8",      # Google DNS
                "1.1.1.1",      # Cloudflare DNS
                "208.67.222.222" # OpenDNS
            ]

            connectivity_results = {}

            for site in test_sites:
                try:
                    # Ping test
                    response = os.system(f"ping -n 1 -w 1000 {site} > nul 2>&1")
                    connectivity_results[site] = response == 0

                except:
                    connectivity_results[site] = False

            # Test internet connectivity
            internet_connected = any(connectivity_results.values())

            return {
                "internet_connected": internet_connected,
                "dns_servers_reachable": connectivity_results,
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"Error checking connectivity: {e}")
            return {"internet_connected": False, "error": str(e)}

    def test_connectivity(self, target: str = "8.8.8.8", timeout: int = 5) -> Dict[str, Any]:
        """
        Test connectivity to a specific target

        Args:
            target: Target host/IP to test
            timeout: Timeout in seconds

        Returns:
            Connectivity test result
        """
        try:
            self.stats["connections_made"] += 1

            # Try to resolve hostname
            try:
                ip = socket.gethostbyname(target)
                dns_success = True
            except socket.gaierror:
                ip = target
                dns_success = False

            # Test connectivity
            start_time = time.time()
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                result = sock.connect_ex((ip, 80))
                sock.close()

                response_time = time.time() - start_time

                if result == 0:
                    return {
                        "success": True,
                        "target": target,
                        "ip": ip,
                        "dns_resolved": dns_success,
                        "response_time": response_time,
                        "port_80_open": True,
                        "timestamp": time.time()
                    }
                else:
                    return {
                        "success": False,
                        "target": target,
                        "ip": ip,
                        "dns_resolved": dns_success,
                        "response_time": response_time,
                        "error": f"Connection failed (error code: {result})",
                        "timestamp": time.time()
                    }

            except socket.timeout:
                return {
                    "success": False,
                    "target": target,
                    "ip": ip,
                    "dns_resolved": dns_success,
                    "error": "Connection timeout",
                    "timestamp": time.time()
                }
            except Exception as e:
                return {
                    "success": False,
                    "target": target,
                    "ip": ip,
                    "dns_resolved": dns_success,
                    "error": str(e),
                    "timestamp": time.time()
                }

        except Exception as e:
            self.stats["connections_failed"] += 1
            self.logger.error(f"Error testing connectivity to {target}: {e}")
            return {
                "success": False,
                "target": target,
                "error": str(e),
                "timestamp": time.time()
            }

    def scan_ports(self,
                   target: str,
                   ports: List[int] = None,
                   timeout: float = 1.0) -> Dict[str, Any]:
        """
        Scan ports on a target host

        Args:
            target: Target host to scan
            ports: List of ports to scan (default common ports)
            timeout: Timeout per port

        Returns:
            Port scan results
        """
        if ports is None:
            # Common ports to scan
            ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 3389]

        try:
            open_ports = []
            closed_ports = []

            # Resolve hostname
            try:
                ip = socket.gethostbyname(target)
            except socket.gaierror:
                ip = target

            for port in ports:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(timeout)
                    result = sock.connect_ex((ip, port))
                    sock.close()

                    if result == 0:
                        open_ports.append(port)
                    else:
                        closed_ports.append(port)

                except Exception as e:
                    self.logger.debug(f"Error scanning port {port}: {e}")
                    closed_ports.append(port)

            return {
                "success": True,
                "target": target,
                "ip": ip,
                "open_ports": open_ports,
                "closed_ports": closed_ports,
                "total_ports": len(ports),
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"Error scanning ports on {target}: {e}")
            return {
                "success": False,
                "target": target,
                "error": str(e),
                "timestamp": time.time()
            }

    def get_network_info(self) -> Dict[str, Any]:
        """Get comprehensive network information"""
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)

            return {
                "hostname": hostname,
                "local_ip": local_ip,
                "interfaces": self.interfaces,
                "default_interface": self.default_interface,
                "gateway": self.gateway,
                "dns_servers": self._get_dns_servers(),
                "current_stats": self.get_network_statistics(),
                "connectivity": self._check_connectivity(),
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"Error getting network info: {e}")
            return {"error": str(e)}

    def _get_dns_servers(self) -> List[str]:
        """Get DNS server addresses"""
        try:
            # This is a simplified implementation
            # In a real system, you might read from registry or system files
            dns_servers = []

            # Try to get from system
            try:
                output = subprocess.check_output("ipconfig /all", shell=True).decode()
                for line in output.split('\n'):
                    if 'DNS Servers' in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            dns_ip = parts[1].strip()
                            if dns_ip and dns_ip not in dns_servers:
                                dns_servers.append(dns_ip)
            except:
                pass

            return dns_servers or ["8.8.8.8", "8.8.4.4"]  # Fallback to Google DNS

        except Exception as e:
            self.logger.error(f"Error getting DNS servers: {e}")
            return []

    def download_file(self,
                     url: str,
                     destination: str = None,
                     progress_callback: callable = None) -> Dict[str, Any]:
        """
        Download file from URL

        Args:
            url: URL to download from
            destination: Local destination path
            progress_callback: Optional progress callback

        Returns:
            Download result
        """
        try:
            if not destination:
                filename = os.path.basename(url.split('?')[0])  # Remove query parameters
                destination = os.path.join(os.path.dirname(__file__), '..', 'data', 'downloads', filename)

            # Ensure destination directory exists
            os.makedirs(os.path.dirname(destination), exist_ok=True)

            # Download file
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0

            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

                        if progress_callback:
                            progress = (downloaded_size / total_size * 100) if total_size > 0 else 0
                            progress_callback(progress, downloaded_size, total_size)

            return {
                "success": True,
                "url": url,
                "destination": destination,
                "file_size": downloaded_size,
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"Error downloading file from {url}: {e}")
            return {
                "success": False,
                "url": url,
                "error": str(e),
                "timestamp": time.time()
            }

    def get_public_ip(self) -> Dict[str, Any]:
        """Get public IP address"""
        try:
            services = [
                "https://api.ipify.org",
                "https://icanhazip.com",
                "https://ipinfo.io/ip"
            ]

            for service in services:
                try:
                    response = requests.get(service, timeout=5)
                    if response.status_code == 200:
                        ip = response.text.strip()
                        return {
                            "success": True,
                            "public_ip": ip,
                            "service": service,
                            "timestamp": time.time()
                        }
                except:
                    continue

            return {
                "success": False,
                "error": "Could not determine public IP",
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"Error getting public IP: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

    def traceroute(self, target: str, max_hops: int = 30) -> Dict[str, Any]:
        """Perform traceroute to target"""
        try:
            # Use system traceroute command
            cmd = f"tracert -h {max_hops} {target}"

            try:
                output = subprocess.check_output(cmd, shell=True, timeout=30).decode()
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "target": target,
                    "error": "Traceroute timeout",
                    "timestamp": time.time()
                }

            # Parse traceroute output (simplified)
            hops = []
            for line in output.split('\n'):
                if line.strip() and not line.startswith('Tracing'):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        hop_info = {
                            "hop": int(parts[0]) if parts[0].isdigit() else 0,
                            "ip": parts[1] if len(parts) > 1 else "",
                            "hostname": " ".join(parts[2:]) if len(parts) > 2 else "",
                            "response_time": parts[0] if not parts[0].isdigit() else ""
                        }
                        hops.append(hop_info)

            return {
                "success": True,
                "target": target,
                "hops": hops,
                "total_hops": len(hops),
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"Error performing traceroute to {target}: {e}")
            return {
                "success": False,
                "target": target,
                "error": str(e),
                "timestamp": time.time()
            }

    def get_wifi_info(self) -> Dict[str, Any]:
        """Get WiFi network information"""
        try:
            # This is Windows-specific
            try:
                output = subprocess.check_output("netsh wlan show interfaces", shell=True).decode()

                wifi_info = {}
                for line in output.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        wifi_info[key.strip()] = value.strip()

                return {
                    "success": True,
                    "wifi_info": wifi_info,
                    "timestamp": time.time()
                }

            except subprocess.CalledProcessError:
                return {
                    "success": False,
                    "error": "No WiFi adapter found or not connected",
                    "timestamp": time.time()
                }

        except Exception as e:
            self.logger.error(f"Error getting WiFi info: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

    def speed_test(self) -> Dict[str, Any]:
        """Perform internet speed test"""
        try:
            self.logger.info("Performing internet speed test...")

            # Use speedtest-cli if available
            try:
                output = subprocess.check_output("speedtest-cli --simple", shell=True, timeout=60).decode()

                # Parse output
                download = upload = ping = 0
                for line in output.split('\n'):
                    if 'Download:' in line:
                        download = float(line.split()[1])
                    elif 'Upload:' in line:
                        upload = float(line.split()[1])
                    elif 'Ping:' in line:
                        ping = float(line.split()[1])

                return {
                    "success": True,
                    "download_mbps": download,
                    "upload_mbps": upload,
                    "ping_ms": ping,
                    "timestamp": time.time()
                }

            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                # Fallback to basic connectivity test
                connectivity = self._check_connectivity()
                return {
                    "success": True,
                    "download_mbps": 0,
                    "upload_mbps": 0,
                    "ping_ms": 0,
                    "connectivity": connectivity,
                    "note": "Speed test requires speedtest-cli",
                    "timestamp": time.time()
                }

        except Exception as e:
            self.logger.error(f"Error performing speed test: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

    def get_active_connections(self) -> List[Dict[str, Any]]:
        """Get list of active network connections"""
        return self.active_connections

    def get_connection_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get connection history"""
        return self.connection_history[-limit:] if self.connection_history else []

    def get_stats(self) -> Dict[str, Any]:
        """Get network manager statistics"""
        return {
            **self.stats,
            "monitoring": self.monitoring,
            "interfaces_count": len(self.interfaces),
            "active_connections": len(self.active_connections),
            "default_interface": self.default_interface
        }

    def clear_history(self):
        """Clear connection history"""
        self.connection_history.clear()
        self.logger.info("Network history cleared")