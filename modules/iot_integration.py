"""
J.A.R.V.I.S. IoT Integration
Smart home and IoT device control and monitoring
"""

import os
import time
import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
import logging

# Import IoT libraries
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

try:
    import bluetooth
    BLUETOOTH_AVAILABLE = True
except ImportError:
    BLUETOOTH_AVAILABLE = False

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    import serial.tools.list_ports
    SERIAL_TOOLS_AVAILABLE = True
except ImportError:
    SERIAL_TOOLS_AVAILABLE = False

try:
    import requests
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False


class IoTDevice:
    """Represents an IoT device"""

    def __init__(self,
                 device_id: str,
                 device_type: str,
                 name: str,
                 connection_info: Dict[str, Any]):
        """
        Initialize IoT device

        Args:
            device_id: Unique device identifier
            device_type: Type of device (light, thermostat, camera, etc.)
            name: Human-readable device name
            connection_info: Connection parameters
        """
        self.device_id = device_id
        self.device_type = device_type
        self.name = name
        self.connection_info = connection_info

        # Device state
        self.is_connected = False
        self.last_seen = None
        self.device_state = {}
        self.capabilities = []

        # Communication
        self.protocol = connection_info.get("protocol", "http")
        self.client = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert device to dictionary"""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "name": self.name,
            "connection_info": self.connection_info,
            "is_connected": self.is_connected,
            "last_seen": self.last_seen,
            "device_state": self.device_state,
            "capabilities": self.capabilities,
            "protocol": self.protocol
        }


class IoTIntegration:
    """
    Advanced IoT integration system
    Connects and controls smart home devices and IoT sensors
    """

    def __init__(self, jarvis_instance):
        """
        Initialize IoT integration

        Args:
            jarvis_instance: Reference to main JARVIS instance
        """
        self.jarvis = jarvis_instance
        self.logger = logging.getLogger('JARVIS.IoTIntegration')

        # Device management
        self.devices: Dict[str, IoTDevice] = {}
        self.device_discovery = {}

        # Communication protocols
        self.mqtt_client = None
        self.mqtt_connected = False

        # Configuration
        self.config = {
            "mqtt_broker": "localhost",
            "mqtt_port": 1883,
            "mqtt_username": None,
            "mqtt_password": None,
            "auto_discovery": True,
            "device_timeout": 300,  # 5 minutes
            "supported_protocols": ["mqtt", "http", "bluetooth", "serial"]
        }

        # Device state tracking
        self.device_states = {}
        self.state_history = []

        # Statistics
        self.stats = {
            "devices_connected": 0,
            "commands_sent": 0,
            "data_received": 0,
            "protocols_active": 0
        }

    async def initialize(self):
        """Initialize IoT integration"""
        try:
            self.logger.info("Initializing IoT integration...")

            # Initialize MQTT client if available
            if MQTT_AVAILABLE:
                await self._initialize_mqtt()

            # Load existing devices
            await self._load_devices()

            # Start device monitoring
            asyncio.create_task(self._device_monitoring_loop())

            # Start auto-discovery if enabled
            if self.config["auto_discovery"]:
                asyncio.create_task(self._auto_discovery_loop())

            self.logger.info("IoT integration initialized")

        except Exception as e:
            self.logger.error(f"Error initializing IoT integration: {e}")
            raise

    async def _initialize_mqtt(self):
        """Initialize MQTT client"""
        try:
            self.mqtt_client = mqtt.Client(client_id="jarvis_iot")

            if self.config["mqtt_username"] and self.config["mqtt_password"]:
                self.mqtt_client.username_pw_set(
                    self.config["mqtt_username"],
                    self.config["mqtt_password"]
                )

            # Set up callbacks
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            self.mqtt_client.on_message = self._on_mqtt_message

            # Connect to broker
            self.mqtt_client.connect_async(
                self.config["mqtt_broker"],
                self.config["mqtt_port"],
                keepalive=60
            )

            # Start MQTT loop
            self.mqtt_client.loop_start()

            self.logger.info("MQTT client initialized")

        except Exception as e:
            self.logger.error(f"Error initializing MQTT: {e}")

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        try:
            if rc == 0:
                self.mqtt_connected = True
                self.logger.info("MQTT connected successfully")

                # Subscribe to device topics
                client.subscribe("homeassistant/+/+/config")
                client.subscribe("jarvis/devices/+/state")
                client.subscribe("jarvis/commands/+")

            else:
                self.logger.error(f"MQTT connection failed with code: {rc}")

        except Exception as e:
            self.logger.error(f"Error in MQTT connect callback: {e}")

    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        self.mqtt_connected = False
        self.logger.warning("MQTT disconnected")

    def _on_mqtt_message(self, client, userdata, message):
        """MQTT message callback"""
        try:
            self.stats["data_received"] += 1

            # Process MQTT message
            asyncio.create_task(self._process_mqtt_message(message))

        except Exception as e:
            self.logger.error(f"Error in MQTT message callback: {e}")

    async def _process_mqtt_message(self, message):
        """Process incoming MQTT message"""
        try:
            topic = message.topic
            payload = message.payload.decode()

            # Handle different topic types
            if "homeassistant" in topic and "config" in topic:
                # Device discovery message
                await self._handle_device_discovery(topic, payload)

            elif "jarvis/devices" in topic and "state" in topic:
                # Device state update
                await self._handle_device_state_update(topic, payload)

            elif "jarvis/commands" in topic:
                # Command response
                await self._handle_command_response(topic, payload)

        except Exception as e:
            self.logger.error(f"Error processing MQTT message: {e}")

    async def _load_devices(self):
        """Load existing IoT devices"""
        try:
            devices_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'iot_devices.json')

            if os.path.exists(devices_file):
                with open(devices_file, 'r') as f:
                    devices_data = json.load(f)

                for device_data in devices_data.get("devices", []):
                    device = IoTDevice(
                        device_data["device_id"],
                        device_data["device_type"],
                        device_data["name"],
                        device_data["connection_info"]
                    )

                    device.is_connected = device_data.get("is_connected", False)
                    device.last_seen = device_data.get("last_seen")
                    device.device_state = device_data.get("device_state", {})
                    device.capabilities = device_data.get("capabilities", [])

                    self.devices[device.device_id] = device

                self.logger.info(f"Loaded {len(self.devices)} IoT devices")

        except Exception as e:
            self.logger.error(f"Error loading IoT devices: {e}")

    async def register_device(self,
                             device_id: str,
                             device_type: str,
                             name: str,
                             connection_info: Dict[str, Any]) -> bool:
        """
        Register new IoT device

        Args:
            device_id: Unique device identifier
            device_type: Type of device
            name: Human-readable name
            connection_info: Connection parameters

        Returns:
            Registration success
        """
        try:
            if device_id in self.devices:
                return False  # Device already exists

            device = IoTDevice(device_id, device_type, name, connection_info)

            # Determine device capabilities based on type
            device.capabilities = self._get_device_capabilities(device_type)

            # Test connection
            connection_success = await self._test_device_connection(device)

            if connection_success:
                device.is_connected = True
                device.last_seen = time.time()

                self.devices[device_id] = device
                await self._save_devices()

                self.logger.info(f"Registered IoT device: {name} ({device_id})")
                return True
            else:
                self.logger.warning(f"Failed to connect to device: {name}")
                return False

        except Exception as e:
            self.logger.error(f"Error registering IoT device: {e}")
            return False

    def _get_device_capabilities(self, device_type: str) -> List[str]:
        """Get device capabilities based on type"""
        capabilities_map = {
            "light": ["turn_on", "turn_off", "set_brightness", "set_color"],
            "thermostat": ["set_temperature", "get_temperature", "set_mode"],
            "camera": ["take_photo", "start_recording", "stop_recording"],
            "sensor": ["get_reading", "get_history"],
            "lock": ["lock", "unlock", "get_status"],
            "switch": ["turn_on", "turn_off", "toggle"],
            "speaker": ["play_audio", "set_volume", "stop_audio"],
            "display": ["show_text", "show_image", "clear_display"]
        }

        return capabilities_map.get(device_type, ["basic_control"])

    async def _test_device_connection(self, device: IoTDevice) -> bool:
        """Test connection to device"""
        try:
            protocol = device.protocol

            if protocol == "mqtt":
                return await self._test_mqtt_connection(device)
            elif protocol == "http":
                return await self._test_http_connection(device)
            elif protocol == "bluetooth":
                return await self._test_bluetooth_connection(device)
            elif protocol == "serial":
                return await self._test_serial_connection(device)
            else:
                return False

        except Exception as e:
            self.logger.error(f"Error testing device connection: {e}")
            return False

    async def _test_mqtt_connection(self, device: IoTDevice) -> bool:
        """Test MQTT device connection"""
        try:
            if not self.mqtt_connected:
                return False

            # Send ping command
            ping_topic = f"jarvis/devices/{device.device_id}/ping"
            self.mqtt_client.publish(ping_topic, "ping")

            # Wait for response (simplified)
            await asyncio.sleep(2)

            return True

        except Exception as e:
            self.logger.error(f"Error testing MQTT connection: {e}")
            return False

    async def _test_http_connection(self, device: IoTDevice) -> bool:
        """Test HTTP device connection"""
        try:
            if not HTTP_AVAILABLE:
                return False

            conn_info = device.connection_info
            base_url = conn_info.get("base_url", "")

            if not base_url:
                return False

            # Test basic connectivity
            response = requests.get(f"{base_url}/status", timeout=5)

            return response.status_code == 200

        except Exception as e:
            self.logger.debug(f"HTTP connection test failed: {e}")
            return False

    async def _test_bluetooth_connection(self, device: IoTDevice) -> bool:
        """Test Bluetooth device connection"""
        try:
            if not BLUETOOTH_AVAILABLE:
                return False

            conn_info = device.connection_info
            mac_address = conn_info.get("mac_address", "")
            port = conn_info.get("port", 1)  # RFCOMM port

            if not mac_address:
                return False

            # Attempt Bluetooth connection
            sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            sock.connect((mac_address, port))
            sock.close()

            return True

        except bluetooth.BluetoothError as e:
            self.logger.debug(f"Bluetooth connection test failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error testing Bluetooth connection: {e}")
            return False

    async def _test_serial_connection(self, device: IoTDevice) -> bool:
        """Test serial device connection"""
        try:
            if not SERIAL_AVAILABLE:
                return False

            conn_info = device.connection_info
            port = conn_info.get("port", "")
            baudrate = conn_info.get("baudrate", 9600)
            timeout = conn_info.get("timeout", 1)

            if not port:
                return False

            # Attempt serial connection
            ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)

            # Try to read a byte to test connection
            ser.write(b'\n')  # Send newline to wake up device
            response = ser.read(1)  # Try to read response

            ser.close()

            # If we get any response or no timeout error, consider it connected
            return True

        except serial.SerialException as e:
            self.logger.debug(f"Serial connection test failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error testing serial connection: {e}")
            return False

    async def control_device(self,
                           device_id: str,
                           command: str,
                           parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Control IoT device

        Args:
            device_id: Device to control
            command: Command to send
            parameters: Command parameters

        Returns:
            Command result
        """
        try:
            if device_id not in self.devices:
                return {
                    "success": False,
                    "error": f"Device not found: {device_id}"
                }

            device = self.devices[device_id]

            if not device.is_connected:
                return {
                    "success": False,
                    "error": f"Device not connected: {device_id}"
                }

            # Execute command based on protocol
            if device.protocol == "mqtt":
                result = await self._send_mqtt_command(device, command, parameters or {})
            elif device.protocol == "http":
                result = await self._send_http_command(device, command, parameters or {})
            elif device.protocol == "bluetooth":
                result = await self._send_bluetooth_command(device, command, parameters or {})
            elif device.protocol == "serial":
                result = await self._send_serial_command(device, command, parameters or {})
            else:
                return {
                    "success": False,
                    "error": f"Unsupported protocol: {device.protocol}"
                }

            self.stats["commands_sent"] += 1

            return result

        except Exception as e:
            self.logger.error(f"Error controlling device {device_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _send_mqtt_command(self, device: IoTDevice, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Send MQTT command to device"""
        try:
            if not self.mqtt_connected:
                return {"success": False, "error": "MQTT not connected"}

            # Build command payload
            payload = {
                "command": command,
                "parameters": parameters,
                "timestamp": time.time()
            }

            # Send command
            topic = f"jarvis/devices/{device.device_id}/command"
            self.mqtt_client.publish(topic, json.dumps(payload))

            return {
                "success": True,
                "message": f"MQTT command sent: {command}",
                "device_id": device.device_id
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _send_http_command(self, device: IoTDevice, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Send HTTP command to device"""
        try:
            if not HTTP_AVAILABLE:
                return {"success": False, "error": "HTTP not available"}

            conn_info = device.connection_info
            base_url = conn_info.get("base_url", "")

            # Build request
            url = f"{base_url}/command"
            payload = {
                "command": command,
                "parameters": parameters
            }

            # Send command
            response = requests.post(url, json=payload, timeout=10)

            if response.status_code == 200:
                return {
                    "success": True,
                    "message": f"HTTP command sent: {command}",
                    "response": response.json()
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP command failed: {response.status_code}"
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _send_bluetooth_command(self, device: IoTDevice, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Send Bluetooth command to device"""
        try:
            if not BLUETOOTH_AVAILABLE:
                return {"success": False, "error": "Bluetooth not available"}

            conn_info = device.connection_info
            mac_address = conn_info.get("mac_address", "")
            port = conn_info.get("port", 1)

            if not mac_address:
                return {"success": False, "error": "No MAC address specified"}

            # Build command payload
            payload = json.dumps({
                "command": command,
                "parameters": parameters,
                "timestamp": time.time()
            }).encode('utf-8')

            # Send via Bluetooth
            sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            sock.connect((mac_address, port))
            sock.send(payload)
            sock.close()

            return {
                "success": True,
                "message": f"Bluetooth command sent: {command}",
                "device_id": device.device_id
            }

        except bluetooth.BluetoothError as e:
            return {"success": False, "error": f"Bluetooth error: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _send_serial_command(self, device: IoTDevice, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Send serial command to device"""
        try:
            if not SERIAL_AVAILABLE:
                return {"success": False, "error": "Serial not available"}

            conn_info = device.connection_info
            port = conn_info.get("port", "")
            baudrate = conn_info.get("baudrate", 9600)
            timeout = conn_info.get("timeout", 1)

            if not port:
                return {"success": False, "error": "No serial port specified"}

            # Build command payload
            payload = json.dumps({
                "command": command,
                "parameters": parameters,
                "timestamp": time.time()
            })

            # Send via serial
            ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
            ser.write((payload + '\n').encode('utf-8'))  # Add newline terminator

            # Try to read response
            response = ser.read(1024)  # Read up to 1KB response
            ser.close()

            return {
                "success": True,
                "message": f"Serial command sent: {command}",
                "device_id": device.device_id,
                "response": response.decode('utf-8', errors='ignore') if response else None
            }

        except serial.SerialException as e:
            return {"success": False, "error": f"Serial error: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_device_discovery(self, topic: str, payload: str):
        """Handle device discovery message"""
        try:
            # Parse Home Assistant discovery message
            discovery_data = json.loads(payload)

            device_id = discovery_data.get("unique_id", "")
            device_name = discovery_data.get("name", "")
            device_type = self._infer_device_type(discovery_data)

            if device_id and device_name:
                # Register discovered device
                await self.register_device(
                    device_id=device_id,
                    device_type=device_type,
                    name=device_name,
                    connection_info={
                        "protocol": "mqtt",
                        "topic": topic,
                        "discovery_data": discovery_data
                    }
                )

        except Exception as e:
            self.logger.error(f"Error handling device discovery: {e}")

    def _infer_device_type(self, discovery_data: Dict[str, Any]) -> str:
        """Infer device type from discovery data"""
        try:
            # Check component type in Home Assistant discovery
            component = discovery_data.get("component", "")

            if component in ["light", "switch", "sensor", "climate", "camera", "lock"]:
                return component

            # Check device class
            device_class = discovery_data.get("device_class", "")

            if device_class in ["temperature", "humidity", "motion"]:
                return "sensor"
            elif device_class in ["door", "window"]:
                return "lock"

            return "unknown"

        except Exception as e:
            self.logger.error(f"Error inferring device type: {e}")
            return "unknown"

    async def _handle_device_state_update(self, topic: str, payload: str):
        """Handle device state update"""
        try:
            # Extract device ID from topic
            topic_parts = topic.split('/')
            if len(topic_parts) >= 3:
                device_id = topic_parts[2]

                if device_id in self.devices:
                    device = self.devices[device_id]

                    # Update device state
                    state_data = json.loads(payload)
                    device.device_state.update(state_data)
                    device.last_seen = time.time()

                    # Store in state history
                    self.state_history.append({
                        "device_id": device_id,
                        "state": state_data,
                        "timestamp": time.time()
                    })

                    # Maintain history size
                    if len(self.state_history) > 10000:
                        self.state_history.pop(0)

        except Exception as e:
            self.logger.error(f"Error handling device state update: {e}")

    async def _handle_command_response(self, topic: str, payload: str):
        """Handle command response"""
        try:
            # Process command response
            response_data = json.loads(payload)

            self.logger.info(f"Command response received: {response_data}")

        except Exception as e:
            self.logger.error(f"Error handling command response: {e}")

    async def _device_monitoring_loop(self):
        """Monitor device connectivity and health"""
        while True:
            try:
                current_time = time.time()

                # Check device connectivity
                for device_id, device in list(self.devices.items()):
                    time_since_last_seen = current_time - (device.last_seen or 0)

                    if time_since_last_seen > self.config["device_timeout"]:
                        device.is_connected = False
                        self.logger.warning(f"Device {device_id} timed out")

                    # Update connection count
                    if device.is_connected:
                        self.stats["devices_connected"] = sum(1 for d in self.devices.values() if d.is_connected)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Error in device monitoring: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    async def _auto_discovery_loop(self):
        """Auto-discovery loop for new devices"""
        while True:
            try:
                if self.mqtt_connected:
                    # Send discovery broadcast
                    discovery_topic = "jarvis/discovery/broadcast"
                    self.mqtt_client.publish(discovery_topic, "discover_devices")

                await asyncio.sleep(300)  # Discover every 5 minutes

            except Exception as e:
                self.logger.error(f"Error in auto-discovery: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes before retry

    async def _save_devices(self):
        """Save devices to persistent storage"""
        try:
            devices_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'iot_devices.json')

            devices_data = {
                "devices": [device.to_dict() for device in self.devices.values()],
                "last_saved": time.time()
            }

            os.makedirs(os.path.dirname(devices_file), exist_ok=True)
            with open(devices_file, 'w') as f:
                json.dump(devices_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving IoT devices: {e}")

    def get_device_info(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a device"""
        if device_id in self.devices:
            return self.devices[device_id].to_dict()
        return None

    def list_devices(self) -> List[Dict[str, Any]]:
        """List all registered devices"""
        return [device.to_dict() for device in self.devices.values()]

    def list_connected_devices(self) -> List[Dict[str, Any]]:
        """List connected devices"""
        return [device.to_dict() for device in self.devices.values() if device.is_connected]

    def get_device_state_history(self, device_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get device state history"""
        device_history = [entry for entry in self.state_history if entry["device_id"] == device_id]
        return device_history[-limit:] if device_history else []

    def get_iot_stats(self) -> Dict[str, Any]:
        """Get IoT integration statistics"""
        return {
            **self.stats,
            "total_devices": len(self.devices),
            "connected_devices": sum(1 for d in self.devices.values() if d.is_connected),
            "protocols_supported": len(self.config["supported_protocols"]),
            "mqtt_connected": self.mqtt_connected
        }

    async def shutdown(self):
        """Shutdown IoT integration"""
        try:
            self.logger.info("Shutting down IoT integration...")

            # Save devices
            await self._save_devices()

            # Close MQTT connection
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()

            self.logger.info("IoT integration shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down IoT integration: {e}")

    # Device control methods

    async def control_light(self, device_id: str, action: str, **kwargs) -> Dict[str, Any]:
        """Control smart light"""
        try:
            if action == "turn_on":
                return await self.control_device(device_id, "turn_on")
            elif action == "turn_off":
                return await self.control_device(device_id, "turn_off")
            elif action == "set_brightness":
                brightness = kwargs.get("brightness", 50)
                return await self.control_device(device_id, "set_brightness", {"brightness": brightness})
            elif action == "set_color":
                color = kwargs.get("color", "#ffffff")
                return await self.control_device(device_id, "set_color", {"color": color})

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def control_thermostat(self, device_id: str, action: str, **kwargs) -> Dict[str, Any]:
        """Control smart thermostat"""
        try:
            if action == "set_temperature":
                temperature = kwargs.get("temperature", 22)
                return await self.control_device(device_id, "set_temperature", {"temperature": temperature})
            elif action == "set_mode":
                mode = kwargs.get("mode", "auto")
                return await self.control_device(device_id, "set_mode", {"mode": mode})

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_sensor_data(self, device_id: str) -> Dict[str, Any]:
        """Get sensor data"""
        try:
            device = self.devices.get(device_id)
            if not device:
                return {"success": False, "error": "Device not found"}

            # Get current state
            return {
                "success": True,
                "device_id": device_id,
                "device_state": device.device_state,
                "last_updated": device.last_seen
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def create_scene(self, scene_name: str, device_actions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create automation scene"""
        try:
            # Store scene configuration
            scene_config = {
                "scene_name": scene_name,
                "device_actions": device_actions,
                "created_at": time.time()
            }

            # Save scene
            scenes_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'scenes.json')

            scenes_data = {}
            if os.path.exists(scenes_file):
                with open(scenes_file, 'r') as f:
                    scenes_data = json.load(f)

            scenes_data[scene_name] = scene_config

            os.makedirs(os.path.dirname(scenes_file), exist_ok=True)
            with open(scenes_file, 'w') as f:
                json.dump(scenes_data, f, indent=2)

            return {
                "success": True,
                "scene_name": scene_name,
                "message": "Scene created successfully"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def activate_scene(self, scene_name: str) -> Dict[str, Any]:
        """Activate automation scene"""
        try:
            scenes_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'scenes.json')

            if not os.path.exists(scenes_file):
                return {"success": False, "error": "No scenes configured"}

            with open(scenes_file, 'r') as f:
                scenes_data = json.load(f)

            if scene_name not in scenes_data:
                return {"success": False, "error": f"Scene not found: {scene_name}"}

            scene_config = scenes_data[scene_name]

            # Execute scene actions
            results = []
            for device_id, action in scene_config["device_actions"].items():
                result = await self.control_device(device_id, action["command"], action.get("parameters", {}))
                results.append(result)

            return {
                "success": True,
                "scene_name": scene_name,
                "results": results
            }

        except Exception as e:
            return {"success": False, "error": str(e)}