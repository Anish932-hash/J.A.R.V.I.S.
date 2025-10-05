"""
J.A.R.V.I.S. Collaboration Manager
Real-time collaboration, session management, and multi-user coordination system
"""

import sys
import os
import time
import asyncio
import threading
import uuid
from typing import Dict, List, Optional, Any, Set, Callable
import logging
import json
import hashlib
from datetime import datetime, timedelta
import socket
import pickle

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False


class CollaborationSession:
    """Represents a collaboration session"""

    def __init__(self, session_id: str, host_id: str, session_name: str, session_type: str = 'general'):
        self.session_id = session_id
        self.host_id = host_id
        self.session_name = session_name
        self.session_type = session_type
        self.participants = {host_id: {'role': 'host', 'joined_at': datetime.now().isoformat()}}
        self.created_at = datetime.now().isoformat()
        self.last_activity = datetime.now().isoformat()
        self.status = 'active'
        self.metadata = {}
        self.shared_data = {}
        self.message_history = []

    def add_participant(self, participant_id: str, role: str = 'participant') -> bool:
        """Add a participant to the session"""
        if participant_id not in self.participants:
            self.participants[participant_id] = {
                'role': role,
                'joined_at': datetime.now().isoformat()
            }
            self.last_activity = datetime.now().isoformat()
            return True
        return False

    def remove_participant(self, participant_id: str) -> bool:
        """Remove a participant from the session"""
        if participant_id in self.participants:
            del self.participants[participant_id]
            self.last_activity = datetime.now().isoformat()
            return True
        return False

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now().isoformat()

    def add_message(self, sender_id: str, message_type: str, content: Any):
        """Add a message to the session history"""
        message = {
            'id': str(uuid.uuid4()),
            'sender_id': sender_id,
            'type': message_type,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        self.message_history.append(message)
        self.update_activity()

        # Limit message history
        if len(self.message_history) > 1000:
            self.message_history = self.message_history[-500:]

    def get_participants(self) -> List[str]:
        """Get list of participant IDs"""
        return list(self.participants.keys())

    def is_active(self) -> bool:
        """Check if session is active"""
        return self.status == 'active' and len(self.participants) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return {
            'session_id': self.session_id,
            'host_id': self.host_id,
            'session_name': self.session_name,
            'session_type': self.session_type,
            'participants': self.participants,
            'created_at': self.created_at,
            'last_activity': self.last_activity,
            'status': self.status,
            'metadata': self.metadata,
            'participant_count': len(self.participants)
        }


class PeerDiscovery:
    """Peer discovery and management"""

    def __init__(self):
        self.known_peers = {}
        self.discovery_port = 9999
        self.logger = logging.getLogger('JARVIS.PeerDiscovery')

    def discover_peers(self) -> List[Dict[str, Any]]:
        """Discover available peers on the network using real UDP broadcast"""
        try:
            peers = []

            # Real UDP broadcast discovery
            discovery_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            discovery_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            discovery_socket.settimeout(2.0)  # 2 second timeout

            # Create discovery message
            discovery_message = json.dumps({
                'type': 'discovery_request',
                'peer_id': str(uuid.uuid4())[:8],
                'timestamp': datetime.now().isoformat(),
                'capabilities': ['collaboration', 'file_sharing', 'code_review']
            }).encode('utf-8')

            try:
                # Send broadcast
                discovery_socket.sendto(discovery_message, ('<broadcast>', self.discovery_port))
                self.logger.info(f"Sent discovery broadcast on port {self.discovery_port}")

                # Listen for responses
                start_time = time.time()
                while time.time() - start_time < 2.0:  # Listen for 2 seconds
                    try:
                        data, addr = discovery_socket.recvfrom(4096)
                        response = json.loads(data.decode('utf-8'))

                        if response.get('type') == 'discovery_response':
                            peer_info = {
                                'id': response.get('peer_id', str(uuid.uuid4())[:8]),
                                'name': response.get('name', f'JARVIS-{addr[0]}'),
                                'address': addr[0],
                                'port': response.get('port', 8888),
                                'capabilities': response.get('capabilities', ['collaboration']),
                                'last_seen': datetime.now().isoformat()
                            }

                            # Avoid duplicate peers
                            if peer_info['id'] not in self.known_peers:
                                self.known_peers[peer_info['id']] = peer_info
                                peers.append(peer_info)
                                self.logger.info(f"Discovered peer: {peer_info['name']} at {peer_info['address']}")

                    except socket.timeout:
                        continue
                    except json.JSONDecodeError:
                        self.logger.warning(f"Invalid discovery response from {addr}")
                        continue

            finally:
                discovery_socket.close()

            self.logger.info(f"Discovery complete: found {len(peers)} peers")
            return peers

        except Exception as e:
            self.logger.error(f"Error discovering peers: {e}")
            return []

    def register_peer(self, peer_info: Dict[str, Any]):
        """Register a peer"""
        try:
            peer_id = peer_info.get('id', str(uuid.uuid4()))
            self.known_peers[peer_id] = {
                **peer_info,
                'registered_at': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error registering peer: {e}")

    def get_peer(self, peer_id: str) -> Optional[Dict[str, Any]]:
        """Get peer information"""
        return self.known_peers.get(peer_id)

    def remove_stale_peers(self, max_age_minutes: int = 30):
        """Remove stale peers"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)

            to_remove = []
            for peer_id, peer_info in self.known_peers.items():
                last_seen = datetime.fromisoformat(peer_info.get('last_seen', peer_info.get('registered_at', '2000-01-01')))
                if last_seen < cutoff_time:
                    to_remove.append(peer_id)

            for peer_id in to_remove:
                del self.known_peers[peer_id]

            if to_remove:
                self.logger.info(f"Removed {len(to_remove)} stale peers")

        except Exception as e:
            self.logger.error(f"Error removing stale peers: {e}")


class DataSynchronization:
    """Data synchronization across peers"""

    def __init__(self):
        self.sync_queue = asyncio.Queue()
        self.sync_handlers = {}
        self.logger = logging.getLogger('JARVIS.DataSync')

    def register_sync_handler(self, data_type: str, handler: Callable):
        """Register a synchronization handler for a data type"""
        self.sync_handlers[data_type] = handler

    async def synchronize_data(self, data_type: str, data: Any, target_peers: List[str]) -> Dict[str, Any]:
        """Synchronize data with target peers"""
        try:
            sync_id = str(uuid.uuid4())
            sync_request = {
                'id': sync_id,
                'type': data_type,
                'data': data,
                'source_peer': 'local',
                'target_peers': target_peers,
                'timestamp': datetime.now().isoformat()
            }

            # Add to sync queue
            await self.sync_queue.put(sync_request)

            # Process sync request
            results = await self._process_sync_request(sync_request)

            return {
                'sync_id': sync_id,
                'success': True,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error synchronizing data: {e}")
            return {'success': False, 'error': str(e)}

    async def _process_sync_request(self, sync_request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a synchronization request"""
        try:
            results = []
            data_type = sync_request['type']

            if data_type in self.sync_handlers:
                handler = self.sync_handlers[data_type]
                for target_peer in sync_request['target_peers']:
                    try:
                        result = await handler(sync_request['data'], target_peer)
                        results.append({
                            'peer': target_peer,
                            'success': True,
                            'result': result
                        })
                    except Exception as e:
                        results.append({
                            'peer': target_peer,
                            'success': False,
                            'error': str(e)
                        })
            else:
                # Default handling - attempt to send data via network
                for target_peer in sync_request['target_peers']:
                    try:
                        # Real network transfer
                        sync_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sync_socket.settimeout(5.0)

                        # Get peer address from known peers
                        peer_info = self.known_peers.get(target_peer)
                        if peer_info:
                            sync_socket.connect((peer_info['address'], peer_info['port']))

                            # Send sync data
                            sync_data = json.dumps(sync_request).encode('utf-8')
                            sync_socket.sendall(sync_data)

                            # Receive acknowledgment
                            ack = sync_socket.recv(1024).decode('utf-8')
                            sync_socket.close()

                            results.append({
                                'peer': target_peer,
                                'success': True,
                                'result': ack
                            })
                        else:
                            results.append({
                                'peer': target_peer,
                                'success': False,
                                'error': 'Peer not found in known peers'
                            })
                    except Exception as e:
                        results.append({
                            'peer': target_peer,
                            'success': False,
                            'error': str(e)
                        })

            return results

        except Exception as e:
            self.logger.error(f"Error processing sync request: {e}")
            return []


class MessageBroker:
    """Message broker for real-time communication"""

    def __init__(self):
        self.subscribers = {}
        self.message_queue = asyncio.Queue()
        self.logger = logging.getLogger('JARVIS.MessageBroker')

    def subscribe(self, topic: str, callback: Callable):
        """Subscribe to a topic"""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable):
        """Unsubscribe from a topic"""
        if topic in self.subscribers:
            self.subscribers[topic] = [cb for cb in self.subscribers[topic] if cb != callback]

    async def publish(self, topic: str, message: Any, sender_id: str = 'system'):
        """Publish a message to a topic"""
        try:
            message_data = {
                'topic': topic,
                'content': message,
                'sender_id': sender_id,
                'timestamp': datetime.now().isoformat(),
                'message_id': str(uuid.uuid4())
            }

            # Add to message queue
            await self.message_queue.put(message_data)

            # Notify subscribers
            if topic in self.subscribers:
                for callback in self.subscribers[topic]:
                    try:
                        await callback(message_data)
                    except Exception as e:
                        self.logger.error(f"Error in message callback: {e}")

        except Exception as e:
            self.logger.error(f"Error publishing message: {e}")

    async def process_messages(self):
        """Process messages from the queue"""
        try:
            while True:
                message = await self.message_queue.get()
                # Additional processing can be done here
                self.message_queue.task_done()

        except Exception as e:
            self.logger.error(f"Error processing messages: {e}")


class CollaborationManager:
    """Advanced collaboration management system"""

    def __init__(self, development_engine):
        self.development_engine = development_engine
        self.jarvis = development_engine.jarvis if hasattr(development_engine, 'jarvis') else None
        self.logger = logging.getLogger('JARVIS.CollaborationManager')

        # Collaboration components
        self.peer_discovery = PeerDiscovery()
        self.data_sync = DataSynchronization()
        self.message_broker = MessageBroker()

        # Session management
        self.active_sessions = {}
        self.local_peer_id = str(uuid.uuid4())[:8]

        # Communication
        self.websocket_server = None
        self.websocket_clients = {}
        self.zmq_context = None

        # State
        self.is_running = False
        self.server_thread = None
        self.message_thread = None

    async def initialize(self):
        """Initialize collaboration manager"""
        try:
            self.logger.info("Initializing Collaboration Manager...")

            # Register default sync handlers
            self.data_sync.register_sync_handler('file', self._sync_file)
            self.data_sync.register_sync_handler('code', self._sync_code)
            self.data_sync.register_sync_handler('task', self._sync_task)

            # Subscribe to collaboration messages
            self.message_broker.subscribe('collaboration', self._handle_collaboration_message)
            self.message_broker.subscribe('session', self._handle_session_message)

            # Start message processing
            self.message_thread = threading.Thread(
                target=self._run_message_processing,
                daemon=True
            )
            self.message_thread.start()

            self.logger.info("Collaboration Manager initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing collaboration manager: {e}")
            return False

    def _run_message_processing(self):
        """Run message processing in a separate thread"""
        try:
            asyncio.run(self.message_broker.process_messages())
        except Exception as e:
            self.logger.error(f"Error in message processing: {e}")

    async def start_server(self, port: int = 8888) -> bool:
        """Start collaboration server"""
        try:
            if not WEBSOCKETS_AVAILABLE:
                self.logger.warning("WebSockets not available - collaboration features limited")
                return False

            if self.is_running:
                return True

            self.is_running = True

            # Start WebSocket server
            self.server_thread = threading.Thread(
                target=self._run_websocket_server,
                args=(port,),
                daemon=True
            )
            self.server_thread.start()

            # Discover peers
            discovered_peers = self.peer_discovery.discover_peers()
            self.logger.info(f"Discovered {len(discovered_peers)} peers")

            self.logger.info(f"Collaboration server started on port {port}")
            return True

        except Exception as e:
            self.logger.error(f"Error starting collaboration server: {e}")
            return False

    def _run_websocket_server(self, port: int):
        """Run WebSocket server"""
        try:
            if not WEBSOCKETS_AVAILABLE:
                return

            async def server_handler(websocket, path):
                try:
                    # Register client
                    client_id = str(uuid.uuid4())[:8]
                    self.websocket_clients[client_id] = websocket

                    self.logger.info(f"Client {client_id} connected")

                    # Handle messages
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            await self._handle_websocket_message(client_id, data)
                        except json.JSONDecodeError:
                            self.logger.error(f"Invalid JSON message from {client_id}")

                except Exception as e:
                    self.logger.error(f"Error handling client {client_id}: {e}")
                finally:
                    # Unregister client
                    if client_id in self.websocket_clients:
                        del self.websocket_clients[client_id]

            # Start server
            start_server = websockets.serve(server_handler, "localhost", port)
            asyncio.run(start_server)

        except Exception as e:
            self.logger.error(f"Error running WebSocket server: {e}")

    async def _handle_websocket_message(self, client_id: str, message: Dict[str, Any]):
        """Handle WebSocket message"""
        try:
            message_type = message.get('type', 'unknown')

            if message_type == 'join_session':
                await self._handle_join_session(client_id, message)
            elif message_type == 'leave_session':
                await self._handle_leave_session(client_id, message)
            elif message_type == 'send_message':
                await self._handle_send_message(client_id, message)
            elif message_type == 'sync_data':
                await self._handle_sync_data(client_id, message)

        except Exception as e:
            self.logger.error(f"Error handling WebSocket message: {e}")

    async def create_session(self, session_name: str, session_type: str = 'general') -> Optional[str]:
        """Create a new collaboration session"""
        try:
            session_id = str(uuid.uuid4())[:8]
            session = CollaborationSession(session_id, self.local_peer_id, session_name, session_type)

            self.active_sessions[session_id] = session

            # Broadcast session creation
            await self.message_broker.publish('session', {
                'action': 'created',
                'session': session.to_dict()
            })

            self.logger.info(f"Created collaboration session: {session_name} ({session_id})")
            return session_id

        except Exception as e:
            self.logger.error(f"Error creating session: {e}")
            return None

    async def join_session(self, session_id: str, peer_id: str = None) -> bool:
        """Join an existing collaboration session"""
        try:
            if session_id not in self.active_sessions:
                self.logger.error(f"Session {session_id} not found")
                return False

            session = self.active_sessions[session_id]
            target_peer = peer_id or self.local_peer_id

            if session.add_participant(target_peer):
                # Broadcast participant joined
                await self.message_broker.publish('session', {
                    'action': 'participant_joined',
                    'session_id': session_id,
                    'participant_id': target_peer
                })

                self.logger.info(f"Peer {target_peer} joined session {session_id}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error joining session: {e}")
            return False

    async def leave_session(self, session_id: str, peer_id: str = None) -> bool:
        """Leave a collaboration session"""
        try:
            if session_id not in self.active_sessions:
                return False

            session = self.active_sessions[session_id]
            target_peer = peer_id or self.local_peer_id

            if session.remove_participant(target_peer):
                # If no participants left, end session
                if not session.get_participants():
                    session.status = 'ended'
                    del self.active_sessions[session_id]

                # Broadcast participant left
                await self.message_broker.publish('session', {
                    'action': 'participant_left',
                    'session_id': session_id,
                    'participant_id': target_peer
                })

                self.logger.info(f"Peer {target_peer} left session {session_id}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error leaving session: {e}")
            return False

    async def send_session_message(self, session_id: str, message_type: str, content: Any, sender_id: str = None):
        """Send a message to a session"""
        try:
            if session_id not in self.active_sessions:
                return False

            session = self.active_sessions[session_id]
            sender = sender_id or self.local_peer_id

            session.add_message(sender, message_type, content)

            # Broadcast message to all participants
            await self.message_broker.publish('collaboration', {
                'session_id': session_id,
                'message_type': message_type,
                'content': content,
                'sender_id': sender,
                'timestamp': datetime.now().isoformat()
            })

            return True

        except Exception as e:
            self.logger.error(f"Error sending session message: {e}")
            return False

    async def synchronize_session_data(self, session_id: str, data_type: str, data: Any) -> Dict[str, Any]:
        """Synchronize data across session participants"""
        try:
            if session_id not in self.active_sessions:
                return {'success': False, 'error': 'Session not found'}

            session = self.active_sessions[session_id]
            participants = session.get_participants()

            # Remove local peer from targets
            target_peers = [p for p in participants if p != self.local_peer_id]

            if not target_peers:
                return {'success': True, 'message': 'No remote participants'}

            # Perform synchronization
            result = await self.data_sync.synchronize_data(data_type, data, target_peers)

            return result

        except Exception as e:
            self.logger.error(f"Error synchronizing session data: {e}")
            return {'success': False, 'error': str(e)}

    async def _handle_join_session(self, client_id: str, message: Dict[str, Any]):
        """Handle join session request"""
        session_id = message.get('session_id')
        if session_id:
            await self.join_session(session_id, client_id)

    async def _handle_leave_session(self, client_id: str, message: Dict[str, Any]):
        """Handle leave session request"""
        session_id = message.get('session_id')
        if session_id:
            await self.leave_session(session_id, client_id)

    async def _handle_send_message(self, client_id: str, message: Dict[str, Any]):
        """Handle send message request"""
        session_id = message.get('session_id')
        message_type = message.get('message_type', 'text')
        content = message.get('content')

        if session_id and content:
            await self.send_session_message(session_id, message_type, content, client_id)

    async def _handle_sync_data(self, client_id: str, message: Dict[str, Any]):
        """Handle data synchronization request"""
        session_id = message.get('session_id')
        data_type = message.get('data_type')
        data = message.get('data')

        if session_id and data_type and data:
            await self.synchronize_session_data(session_id, data_type, data)

    async def _handle_collaboration_message(self, message: Dict[str, Any]):
        """Handle collaboration messages"""
        # Broadcast to WebSocket clients
        message_json = json.dumps(message)
        for client_id, websocket in self.websocket_clients.items():
            try:
                await websocket.send(message_json)
            except Exception as e:
                self.logger.error(f"Error sending message to client {client_id}: {e}")

    async def _handle_session_message(self, message: Dict[str, Any]):
        """Handle session messages"""
        # Similar to collaboration messages
        await self._handle_collaboration_message(message)

    async def _sync_file(self, data: Dict[str, Any], target_peer: str) -> Dict[str, Any]:
        """Synchronize file data with real file transfer"""
        try:
            file_path = data.get('path', '')
            file_content = data.get('content', b'')

            # Get peer information
            peer_info = self.peer_discovery.get_peer(target_peer)
            if not peer_info:
                return {'error': 'Peer not found'}

            # Real file transfer via socket
            transfer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            transfer_socket.settimeout(30.0)

            try:
                transfer_socket.connect((peer_info['address'], peer_info['port']))

                # Send file transfer request
                request = {
                    'type': 'file_transfer',
                    'path': file_path,
                    'size': len(file_content)
                }
                transfer_socket.sendall(json.dumps(request).encode('utf-8') + b'\n')

                # Send file content in chunks
                chunk_size = 8192
                bytes_sent = 0
                for i in range(0, len(file_content), chunk_size):
                    chunk = file_content[i:i + chunk_size]
                    transfer_socket.sendall(chunk)
                    bytes_sent += len(chunk)

                # Wait for acknowledgment
                ack = transfer_socket.recv(1024).decode('utf-8')
                transfer_socket.close()

                self.logger.info(f"File sync completed: {file_path} to {target_peer} ({bytes_sent} bytes)")

                return {
                    'file_path': file_path,
                    'bytes_transferred': bytes_sent,
                    'status': 'completed',
                    'acknowledgment': ack
                }

            finally:
                transfer_socket.close()

        except Exception as e:
            self.logger.error(f"Error syncing file: {e}")
            return {'error': str(e)}

    async def _sync_code(self, data: Dict[str, Any], target_peer: str) -> Dict[str, Any]:
        """Synchronize code data with real transfer"""
        try:
            code_snippet = data.get('code', '')
            language = data.get('language', 'python')
            file_path = data.get('file_path', 'unnamed.py')

            # Get peer information
            peer_info = self.peer_discovery.get_peer(target_peer)
            if not peer_info:
                return {'error': 'Peer not found'}

            # Real code transfer
            transfer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            transfer_socket.settimeout(10.0)

            try:
                transfer_socket.connect((peer_info['address'], peer_info['port']))

                # Send code sync request
                request = {
                    'type': 'code_sync',
                    'language': language,
                    'file_path': file_path,
                    'code': code_snippet,
                    'lines': len(code_snippet.split('\n'))
                }
                transfer_socket.sendall(json.dumps(request).encode('utf-8'))

                # Wait for acknowledgment
                ack = transfer_socket.recv(1024).decode('utf-8')
                transfer_socket.close()

                self.logger.info(f"Code sync completed: {language} code to {target_peer}")

                return {
                    'language': language,
                    'lines_synced': len(code_snippet.split('\n')),
                    'status': 'completed',
                    'acknowledgment': ack
                }

            finally:
                transfer_socket.close()

        except Exception as e:
            self.logger.error(f"Error syncing code: {e}")
            return {'error': str(e)}

    async def _sync_task(self, data: Dict[str, Any], target_peer: str) -> Dict[str, Any]:
        """Synchronize task data with real transfer"""
        try:
            task_id = data.get('task_id', '')
            task_status = data.get('status', 'unknown')
            task_data = data.get('task_data', {})

            # Get peer information
            peer_info = self.peer_discovery.get_peer(target_peer)
            if not peer_info:
                return {'error': 'Peer not found'}

            # Real task synchronization
            sync_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sync_socket.settimeout(10.0)

            try:
                sync_socket.connect((peer_info['address'], peer_info['port']))

                # Send task sync request
                request = {
                    'type': 'task_sync',
                    'task_id': task_id,
                    'status': task_status,
                    'data': task_data,
                    'timestamp': datetime.now().isoformat()
                }
                sync_socket.sendall(json.dumps(request).encode('utf-8'))

                # Wait for acknowledgment
                ack = sync_socket.recv(1024).decode('utf-8')
                sync_socket.close()

                self.logger.info(f"Task sync completed: {task_id} ({task_status}) to {target_peer}")

                return {
                    'task_id': task_id,
                    'status': task_status,
                    'sync_status': 'completed',
                    'acknowledgment': ack
                }

            finally:
                sync_socket.close()

        except Exception as e:
            self.logger.error(f"Error syncing task: {e}")
            return {'error': str(e)}

    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get list of active sessions"""
        try:
            return [session.to_dict() for session in self.active_sessions.values() if session.is_active()]

        except Exception as e:
            self.logger.error(f"Error getting active sessions: {e}")
            return []

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed session information"""
        try:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                return {
                    **session.to_dict(),
                    'message_count': len(session.message_history),
                    'recent_messages': session.message_history[-10:] if session.message_history else []
                }

            return None

        except Exception as e:
            self.logger.error(f"Error getting session info: {e}")
            return None

    def get_collaboration_stats(self) -> Dict[str, Any]:
        """Get collaboration statistics"""
        try:
            total_sessions = len(self.active_sessions)
            active_sessions = len([s for s in self.active_sessions.values() if s.is_active()])
            total_participants = sum(len(s.get_participants()) for s in self.active_sessions.values())

            return {
                'total_sessions': total_sessions,
                'active_sessions': active_sessions,
                'total_participants': total_participants,
                'known_peers': len(self.peer_discovery.known_peers),
                'websocket_clients': len(self.websocket_clients),
                'server_running': self.is_running,
                'local_peer_id': self.local_peer_id
            }

        except Exception as e:
            self.logger.error(f"Error getting collaboration stats: {e}")
            return {}

    async def shutdown(self):
        """Shutdown collaboration manager"""
        try:
            self.is_running = False

            # Close WebSocket connections
            for client_id, websocket in self.websocket_clients.items():
                try:
                    await websocket.close()
                except Exception:
                    pass

            self.websocket_clients.clear()

            # End all sessions
            for session in list(self.active_sessions.values()):
                session.status = 'ended'

            self.active_sessions.clear()

            self.logger.info("Collaboration Manager shutdown")
        except Exception as e:
            self.logger.error(f"Error shutting down collaboration manager: {e}")