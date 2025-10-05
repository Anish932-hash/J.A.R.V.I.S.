"""
J.A.R.V.I.S. Distributed Computing
Support for distributed processing and cluster management
"""

import os
import time
import json
import asyncio
import socket
from typing import Dict, List, Optional, Any
import logging


class DistributedComputing:
    """
    Distributed computing support for J.A.R.V.I.S.
    """

    def __init__(self, jarvis_instance):
        self.jarvis = jarvis_instance
        self.logger = logging.getLogger('JARVIS.DistributedComputing')

        # Node management
        self.local_node = None
        self.cluster_nodes = {}
        self.node_status = {}

        # Task distribution
        self.distributed_tasks = {}
        self.task_results = {}

        # Configuration
        self.config = {
            "cluster_enabled": False,
            "node_discovery_port": 9999,
            "master_node": None,
            "auto_discovery": True,
            "heartbeat_interval": 30
        }

    async def initialize(self):
        """Initialize distributed computing"""
        try:
            self.logger.info("Initializing distributed computing...")

            # Initialize local node
            self.local_node = await self._create_local_node()

            # Start node discovery if enabled
            if self.config["auto_discovery"]:
                asyncio.create_task(self._node_discovery_loop())

            # Start heartbeat if master
            if self.config["master_node"] == self.local_node["node_id"]:
                asyncio.create_task(self._master_heartbeat_loop())

            self.logger.info("Distributed computing initialized")

        except Exception as e:
            self.logger.error(f"Error initializing distributed computing: {e}")
            raise

    async def _create_local_node(self) -> Dict[str, Any]:
        """Create local node information"""
        try:
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)

            node = {
                "node_id": f"jarvis_node_{hostname}_{int(time.time())}",
                "hostname": hostname,
                "ip_address": ip_address,
                "port": self.config["node_discovery_port"],
                "node_type": "worker",  # or "master"
                "capabilities": self._get_node_capabilities(),
                "status": "online",
                "last_seen": time.time(),
                "workload": 0.0,
                "available_memory": self._get_available_memory(),
                "cpu_cores": os.cpu_count() or 1
            }

            return node

        except Exception as e:
            self.logger.error(f"Error creating local node: {e}")
            return {}

    def _get_node_capabilities(self) -> List[str]:
        """Get node processing capabilities"""
        capabilities = ["basic_processing"]

        # Check for GPU
        try:
            import GPUtil
            if len(GPUtil.getGPUs()) > 0:
                capabilities.append("gpu_processing")
        except:
            pass

        # Check for special hardware
        if os.path.exists("/dev/nvidia"):
            capabilities.append("cuda_support")

        return capabilities

    def _get_available_memory(self) -> int:
        """Get available memory in MB"""
        try:
            import psutil
            return int(psutil.virtual_memory().available / (1024 * 1024))
        except:
            return 1024  # Default 1GB

    async def _node_discovery_loop(self):
        """Node discovery loop"""
        while True:
            try:
                # Broadcast discovery message
                await self._broadcast_discovery()

                # Listen for other nodes
                await self._listen_for_nodes()

                await asyncio.sleep(60)  # Discover every minute

            except Exception as e:
                self.logger.error(f"Error in node discovery: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    async def _broadcast_discovery(self):
        """Broadcast node discovery message"""
        try:
            # Create UDP socket for broadcasting
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

            discovery_message = {
                "type": "node_discovery",
                "node_info": self.local_node,
                "timestamp": time.time()
            }

            # Broadcast to network
            sock.sendto(
                json.dumps(discovery_message).encode(),
                ('<broadcast>', self.config["node_discovery_port"])
            )

            sock.close()

        except Exception as e:
            self.logger.error(f"Error broadcasting discovery: {e}")

    async def _listen_for_nodes(self):
        """Listen for other nodes"""
        try:
            # Create UDP socket for listening
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('', self.config["node_discovery_port"]))

            # Set timeout
            sock.settimeout(5.0)

            try:
                while True:
                    data, addr = sock.recvfrom(1024)
                    message = json.loads(data.decode())

                    if message.get("type") == "node_discovery":
                        node_info = message.get("node_info", {})

                        # Add to cluster nodes
                        node_id = node_info.get("node_id", "")
                        if node_id and node_id != self.local_node["node_id"]:
                            self.cluster_nodes[node_id] = node_info
                            self.node_status[node_id] = "online"

            except socket.timeout:
                pass
            finally:
                sock.close()

        except Exception as e:
            self.logger.error(f"Error listening for nodes: {e}")

    async def _master_heartbeat_loop(self):
        """Master node heartbeat loop"""
        while True:
            try:
                # Send heartbeat to all nodes
                for node_id, node_info in self.cluster_nodes.items():
                    await self._send_heartbeat(node_id, node_info)

                await asyncio.sleep(self.config["heartbeat_interval"])

            except Exception as e:
                self.logger.error(f"Error in master heartbeat: {e}")
                await asyncio.sleep(60)

    async def _send_heartbeat(self, node_id: str, node_info: Dict[str, Any]):
        """Send heartbeat to node"""
        try:
            # Create TCP connection to node
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)

            try:
                sock.connect((node_info["ip_address"], node_info["port"]))

                heartbeat_message = {
                    "type": "heartbeat",
                    "master_id": self.local_node["node_id"],
                    "timestamp": time.time()
                }

                sock.send(json.dumps(heartbeat_message).encode())

                # Wait for response
                response = sock.recv(1024)
                response_data = json.loads(response.decode())

                if response_data.get("status") == "alive":
                    self.node_status[node_id] = "online"
                    node_info["last_seen"] = time.time()

            except (ConnectionRefusedError, socket.timeout):
                self.node_status[node_id] = "offline"
            finally:
                sock.close()

        except Exception as e:
            self.logger.error(f"Error sending heartbeat to {node_id}: {e}")

    async def distribute_task(self, task_data: Dict[str, Any]) -> str:
        """Distribute task to cluster"""
        try:
            task_id = f"distributed_task_{int(time.time())}"

            # Find best node for task
            target_node = await self._select_best_node(task_data)

            if target_node:
                # Send task to node
                await self._send_task_to_node(target_node, task_id, task_data)

                self.distributed_tasks[task_id] = {
                    "task_id": task_id,
                    "target_node": target_node,
                    "task_data": task_data,
                    "status": "sent",
                    "created_at": time.time()
                }

                return task_id
            else:
                return ""

        except Exception as e:
            self.logger.error(f"Error distributing task: {e}")
            return ""

    async def _select_best_node(self, task_data: Dict[str, Any]) -> Optional[str]:
        """Select best node for task"""
        try:
            best_node = None
            best_score = -1

            for node_id, node_info in self.cluster_nodes.items():
                if self.node_status.get(node_id) != "online":
                    continue

                # Calculate node score
                score = self._calculate_node_score(node_info, task_data)

                if score > best_score:
                    best_score = score
                    best_node = node_id

            return best_node

        except Exception as e:
            self.logger.error(f"Error selecting best node: {e}")
            return None

    def _calculate_node_score(self, node_info: Dict[str, Any], task_data: Dict[str, Any]) -> float:
        """Calculate score for node suitability"""
        try:
            score = 0.0

            # Base score from workload (lower workload = higher score)
            workload = node_info.get("workload", 0)
            score += (1 - workload) * 50

            # Memory availability
            available_memory = node_info.get("available_memory", 0)
            memory_requirement = task_data.get("memory_requirement", 100)
            if available_memory >= memory_requirement:
                score += 30
            else:
                score += (available_memory / memory_requirement) * 30

            # Capability match
            node_capabilities = set(node_info.get("capabilities", []))
            task_requirements = set(task_data.get("requirements", []))

            capability_match = len(node_capabilities & task_requirements) / len(task_requirements) if task_requirements else 1
            score += capability_match * 20

            return score

        except Exception as e:
            self.logger.error(f"Error calculating node score: {e}")
            return 0.0

    async def _send_task_to_node(self, node_id: str, task_id: str, task_data: Dict[str, Any]):
        """Send task to specific node"""
        try:
            node_info = self.cluster_nodes[node_id]

            # Create TCP connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10.0)

            try:
                sock.connect((node_info["ip_address"], node_info["port"]))

                task_message = {
                    "type": "task",
                    "task_id": task_id,
                    "task_data": task_data,
                    "source_node": self.local_node["node_id"]
                }

                sock.send(json.dumps(task_message).encode())

            finally:
                sock.close()

        except Exception as e:
            self.logger.error(f"Error sending task to node {node_id}: {e}")

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster status"""
        return {
            "local_node": self.local_node,
            "cluster_nodes": list(self.cluster_nodes.keys()),
            "online_nodes": [nid for nid, status in self.node_status.items() if status == "online"],
            "offline_nodes": [nid for nid, status in self.node_status.items() if status == "offline"],
            "distributed_tasks": len(self.distributed_tasks),
            "cluster_enabled": self.config["cluster_enabled"]
        }

    async def shutdown(self):
        """Shutdown distributed computing"""
        try:
            self.logger.info("Shutting down distributed computing...")

            # Notify other nodes
            await self._broadcast_shutdown()

            self.logger.info("Distributed computing shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down distributed computing: {e}")

    async def _broadcast_shutdown(self):
        """Broadcast shutdown message"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

            shutdown_message = {
                "type": "node_shutdown",
                "node_id": self.local_node["node_id"],
                "timestamp": time.time()
            }

            sock.sendto(
                json.dumps(shutdown_message).encode(),
                ('<broadcast>', self.config["node_discovery_port"])
            )

            sock.close()

        except Exception as e:
            self.logger.error(f"Error broadcasting shutdown: {e}")