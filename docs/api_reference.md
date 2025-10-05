# J.A.R.V.I.S. API Reference

## Overview

J.A.R.V.I.S. 2.0 provides a comprehensive API for integrating with external systems and extending functionality. This document covers all public APIs and interfaces.

## Core API

### JARVIS Main Class

```python
from jarvis import JARVIS

# Initialize JARVIS
jarvis = JARVIS(config_path="path/to/config.json")

# Start system
jarvis.start()

# Execute commands
result = jarvis.execute_command("status")

# Get system status
status = jarvis.get_status()

# Speak text
jarvis.speak("Hello, world!")

# Shutdown
jarvis.shutdown()
```

### System Status API

```python
status = jarvis.get_status()
# Returns:
{
    "is_running": bool,
    "status": str,
    "uptime": float,
    "performance_metrics": dict,
    "active_modules": list,
    "version": str
}
```

## Module APIs

### Voice Interface API

```python
from jarvis.modules.voice_interface import VoiceInterface

voice = VoiceInterface(jarvis)

# Initialize
voice.initialize()

# Text to speech
voice.speak("Hello!", priority="normal")

# Speech recognition
text = voice.listen(timeout=10)

# Continuous listening
voice.start_continuous_listening()
voice.stop_continuous_listening()

# Voice settings
voice.set_voice("english")
voice.set_speech_rate(200)
voice.set_speech_volume(0.8)

# Get available voices
voices = voice.get_available_voices()

# Test functionality
voice.test_speech()
voice.test_listening()
```

### System Monitor API

```python
from jarvis.modules.system_monitor import SystemMonitor

monitor = SystemMonitor(jarvis)
monitor.start_monitoring()

# Get current readings
readings = monitor.get_current_readings()

# Get health status
health = monitor.get_health_status()

# Get system info
info = monitor.get_system_info()

# Get top processes
processes = monitor.get_top_processes(limit=10)

# Run speed test
speed_test = monitor.run_speed_test()

# Kill process
monitor.kill_process(pid)

# Set thresholds
monitor.set_threshold("cpu_percent", 80)
```

### Application Controller API

```python
from jarvis.modules.application_controller import ApplicationController

app_controller = ApplicationController(jarvis)
app_controller.initialize()

# Launch application
result = app_controller.launch_application("notepad.exe", arguments="file.txt")

# Close application
result = app_controller.close_application("notepad.exe")

# Switch to application
result = app_controller.switch_to_application("chrome.exe")

# Get application info
info = app_controller.get_application_info("chrome.exe")

# List running applications
apps = app_controller.list_running_applications()

# Search applications
results = app_controller.search_applications("chrome")

# Window manipulation
app_controller.minimize_window(hwnd)
app_controller.maximize_window(hwnd)
app_controller.resize_window(hwnd, 800, 600, 100, 100)
```

### File Manager API

```python
from jarvis.modules.file_manager import FileManager

file_manager = FileManager(jarvis)
file_manager.initialize()

# Create file
result = file_manager.create_file("test.txt", "Hello, world!")

# Delete file
result = file_manager.delete_file("test.txt")

# Copy file
result = file_manager.copy_file("source.txt", "dest.txt")

# Move file
result = file_manager.move_file("source.txt", "new_dest.txt")

# Search files
results = file_manager.search_files("query", "/path/to/search")

# List directory
contents = file_manager.list_directory("/path")

# Create directory
result = file_manager.create_directory("/new/folder")

# Compress files
result = file_manager.compress_files(["file1.txt", "file2.txt"], "archive.zip")

# Extract archive
result = file_manager.extract_archive("archive.zip", "/extract/to")

# Find duplicates
duplicates = file_manager.find_duplicates("/folder")

# Organize files
result = file_manager.organize_files("/source", "/organized", "type")

# Get disk usage
usage = file_manager.get_disk_usage("/")

# Cleanup temp files
result = file_manager.cleanup_temp_files(older_than_days=7)
```

### Network Manager API

```python
from jarvis.modules.network_manager import NetworkManager

network_manager = NetworkManager(jarvis)
network_manager.initialize()

# Get network info
info = network_manager.get_network_info()

# Test connectivity
result = network_manager.test_connectivity("8.8.8.8")

# Scan ports
result = network_manager.scan_ports("example.com", [80, 443])

# Get public IP
ip_result = network_manager.get_public_ip()

# Speed test
speed_result = network_manager.speed_test()

# Download file
result = network_manager.download_file("http://example.com/file.zip")

# Traceroute
trace_result = network_manager.traceroute("example.com")

# WiFi info
wifi_info = network_manager.get_wifi_info()

# Get active connections
connections = network_manager.get_active_connections()
```

### Security Manager API

```python
from jarvis.modules.security_manager import SecurityManager

security_manager = SecurityManager(jarvis)
security_manager.initialize()

# Authenticate user
result = security_manager.authenticate_user("username", "password")

# Validate session
is_valid = security_manager.validate_session("username", "session_token")

# Logout user
security_manager.logout_user("username")

# Encrypt data
encrypted = security_manager.encrypt_data("sensitive data")

# Decrypt data
decrypted = security_manager.decrypt_data(encrypted)

# Check permissions
has_permission = security_manager.check_permission("user", "resource", "action")

# Grant permission
security_manager.grant_permission("user", "resource", "action", True)

# Scan for malware
scan_result = security_manager.scan_for_malware("/path")

# Get firewall status
firewall_status = security_manager.get_firewall_status()

# Get security audit log
audit_log = security_manager.get_security_audit_log(limit=100)
```

## Advanced APIs

### API Manager API

```python
from jarvis.core.api_manager import APIManager

api_manager = APIManager(jarvis)

# Initialize
await api_manager.initialize()

# Create API request
request = api_manager.APIRequest(
    provider=api_manager.APIProvider.OPENAI,
    model="gpt-4",
    prompt="Hello, world!",
    request_type="text"
)

# Make request
response = await api_manager.make_request(request)

# Get provider info
info = api_manager.get_provider_info(api_manager.APIProvider.OPENAI)

# Get all providers
providers = api_manager.get_all_providers()

# Get statistics
stats = api_manager.get_stats()
```

### Self-Development Engine API

```python
from jarvis.core.advanced.self_development_engine import SelfDevelopmentEngine

dev_engine = SelfDevelopmentEngine(jarvis)

# Initialize
await dev_engine.initialize()

# Create development task
task_id = await dev_engine.create_task(
    task_type="feature",
    description="Add image recognition feature",
    priority=8,
    requirements={"accuracy": 0.95}
)

# Get task status
status = dev_engine.get_task_status(task_id)

# Get all tasks
tasks = dev_engine.get_all_tasks()

# Get statistics
stats = dev_engine.get_stats()
```

### Application Healer API

```python
from jarvis.core.advanced.application_healer import ApplicationHealer

healer = ApplicationHealer(jarvis)

# Initialize
await healer.initialize()

# Trigger manual healing
task_id = await healer.trigger_manual_healing("system")

# Get healing status
status = healer.get_healing_status()

# Get healing history
history = healer.get_healing_history(limit=50)

# Enable/disable auto-healing
healer.enable_auto_healing()
healer.disable_auto_healing()
```

### Memory Manager API

```python
from jarvis.core.advanced.memory_manager import MemoryManager

memory_manager = MemoryManager(jarvis)

# Initialize
await memory_manager.initialize()

# Store memory
memory_id = await memory_manager.store_memory(
    content="Important information",
    memory_type="fact",
    importance=0.8
)

# Retrieve memories
memories = await memory_manager.retrieve_memories(
    query="important",
    limit=10,
    memory_type="fact"
)

# Search similar memories
similar = await memory_manager.search_similar_memories("query", limit=5)

# Get memory context
context = await memory_manager.get_memory_context("current query")

# Update memory importance
await memory_manager.update_memory_importance(memory_id, 0.9)

# Forget memories
await memory_manager.forget_memories({"older_than_days": 30})

# Export/Import memories
await memory_manager.export_memories("memories_backup.json")
await memory_manager.import_memories("memories_backup.json")
```

### Ethics Engine API

```python
from jarvis.core.advanced.ethics_engine import EthicsEngine

ethics_engine = EthicsEngine(jarvis)

# Initialize
await ethics_engine.initialize()

# Check response ethics
result = await ethics_engine.check_response("Response text", context)

# Perform ethics audit
audit = await ethics_engine.audit_response("Response text", context)

# Get ethics status
status = ethics_engine.get_ethics_status()

# Update guidelines
ethics_engine.update_guideline("harmful_content", "enabled", True)

# Add blocked topic
ethics_engine.add_blocked_topic("inappropriate topic")

# Generate ethics report
report = ethics_engine.generate_ethics_report()
```

## GUI APIs

### Advanced GUI API

```python
from jarvis.gui.advanced_gui import AdvancedJARVISGUI, create_advanced_gui

# Create GUI
main_window, app = create_advanced_gui(jarvis)

if main_window:
    main_window.show()
    app.exec_()
```

### Main Window API

```python
from jarvis.gui.main_window import MainWindow

gui = MainWindow(jarvis)
gui.create_window()
gui.run()
```

## Plugin API

### Plugin Manager API

```python
from jarvis.modules.plugin_manager import PluginManager

plugin_manager = PluginManager(jarvis)

# Load plugins
plugin_manager.load_plugins()

# Enable plugin
plugin_manager.enable_plugin("my_plugin")

# Disable plugin
plugin_manager.disable_plugin("my_plugin")

# Get plugin info
info = plugin_manager.get_plugin_info("my_plugin")

# List plugins
plugins = plugin_manager.list_plugins()

# Create plugin skeleton
plugin_manager.create_plugin_skeleton("MyPlugin")
```

### Custom Plugin Development

```python
from jarvis.modules.plugin_manager import Plugin

class MyPlugin(Plugin):
    def __init__(self):
        super().__init__(
            name="MyPlugin",
            version="1.0.0",
            description="My custom plugin",
            author="Developer Name"
        )

        self.commands = [
            {
                "name": "my_command",
                "pattern": r"do (.+)",
                "entities": ["action"],
                "handler": self.handle_command,
                "description": "My command description"
            }
        ]

        self.hooks = {
            "on_command": self.on_command,
            "on_system_startup": self.on_startup
        }

    def handle_command(self, command):
        # Handle command logic
        return {
            "action": "my_command",
            "message": "Command executed successfully"
        }

    def on_command(self, command):
        # Hook into command processing
        pass

    def on_startup(self):
        # Called on system startup
        pass
```

## Web APIs

### Web Searcher API

```python
from jarvis.modules.web_searcher import WebSearcher

web_searcher = WebSearcher(jarvis)

# Initialize
await web_searcher.initialize()

# Search web
results = await web_searcher.search("AI research", max_results=20)

# Search GitHub
github_results = await web_searcher.search_github("machine learning", "python")

# Search Stack Overflow
so_results = await web_searcher.search_stackoverflow("python error")

# Search arXiv
arxiv_results = await web_searcher.search_arxiv("neural networks")

# Get page content
content = await web_searcher.get_page_content("https://example.com")
```

### Multimedia Processor API

```python
from jarvis.modules.multimedia_processor import MultimediaProcessor

processor = MultimediaProcessor(jarvis)

# Initialize
await processor.initialize()

# Process image
result = await processor.process_image("input.jpg", ["auto_enhance"], "output.jpg")

# Analyze image
analysis = await processor.analyze_image("image.jpg")

# Generate caption
caption = await processor.generate_image_caption("image.jpg")

# Process video
result = await processor.process_video("input.mp4", ["stabilize"], "output.mp4")

# Extract audio
result = await processor.extract_audio_from_video("video.mp4", "audio.wav")

# Create thumbnail
result = await processor.create_video_thumbnail("video.mp4", "thumb.jpg")

# Batch process
result = await processor.batch_process_images("/input", ["enhance"], "/output")
```

### Automation Engine API

```python
from jarvis.modules.automation_engine import AutomationEngine

automation = AutomationEngine(jarvis)

# Initialize
await automation.initialize()

# Create automation task
task_id = await automation.create_task(
    "desktop",
    "Open calculator",
    [{"type": "click", "position": (100, 100)}]
)

# Automate web task
web_task_id = await automation.automate_web_task(
    "https://example.com",
    [
        {"type": "navigate", "url": "https://example.com"},
        {"type": "click", "selector": "#button", "by": "css"}
    ]
)

# Get screen size
width, height = automation.get_screen_size()

# Take screenshot
screenshot_path = automation.take_screenshot("screenshot.png")

# Get mouse position
x, y = automation.get_mouse_position()
```

## Utility APIs

### Configuration API

```python
from jarvis.utils.config import ConfigManager

config = ConfigManager()

# Load configuration
config.load("config.json")

# Get setting
value = config.get("section.setting")

# Set setting
config.set("section.setting", value)

# Save configuration
config.save("config.json")
```

### Logger API

```python
from jarvis.utils.logger import AdvancedLogger

logger = AdvancedLogger("MyModule")

# Log messages
logger.info("Information message")
logger.warning("Warning message")
logger.error("Error message")
logger.debug("Debug message")

# Get logs
recent_logs = logger.get_recent_logs(100)
```

## Integration Examples

### Discord Bot Integration

```python
import discord
from jarvis import JARVIS

class JARVISDiscordBot(discord.Client):
    def __init__(self):
        super().__init__()
        self.jarvis = JARVIS()

    async def on_message(self, message):
        if message.content.startswith('!jarvis'):
            command = message.content[7:].strip()
            result = self.jarvis.execute_command(command)

            await message.channel.send(f"JARVIS: {result}")
```

### Web Interface Integration

```python
from flask import Flask
from jarvis import JARVIS

app = Flask(__name__)
jarvis = JARVIS()

@app.route('/command/<command>')
def execute_command(command):
    result = jarvis.execute_command(command)
    return {"result": result}

@app.route('/status')
def get_status():
    return jarvis.get_status()

if __name__ == '__main__':
    jarvis.start()
    app.run(host='0.0.0.0', port=8080)
```

### IoT Device Integration

```python
import paho.mqtt.client as mqtt
from jarvis import JARVIS

class IoTIntegration:
    def __init__(self, jarvis):
        self.jarvis = jarvis
        self.client = mqtt.Client()

    def on_message(self, client, userdata, message):
        # Process IoT data
        data = json.loads(message.payload.decode())
        self.jarvis.execute_command(f"process_iot_data {data}")

    def start(self):
        self.client.on_message = self.on_message
        self.client.connect("iot-broker")
        self.client.subscribe("sensors/#")
        self.client.loop_start()
```

## Error Handling

All APIs follow consistent error handling patterns:

```python
try:
    result = jarvis.execute_command("some_command")
    if result.get("success", False):
        # Success handling
        pass
    else:
        # Error handling
        error = result.get("error", "Unknown error")
except Exception as e:
    # Exception handling
    logger.error(f"API error: {e}")
```

## Rate Limiting

APIs implement rate limiting to prevent abuse:

```python
# Check rate limits before making requests
if api_manager.check_rate_limit(provider):
    response = await api_manager.make_request(request)
else:
    # Handle rate limit exceeded
    pass
```

## Authentication

For secure APIs:

```python
# Authenticate before using protected features
auth_result = security_manager.authenticate_user("username", "password")
if auth_result["success"]:
    session_token = auth_result["session_token"]
    # Use token for subsequent requests
else:
    # Handle authentication failure
    pass
```

## Best Practices

1. **Error Handling**: Always check for errors in API responses
2. **Resource Management**: Properly initialize and shutdown components
3. **Rate Limiting**: Respect API rate limits
4. **Security**: Use encryption for sensitive data
5. **Logging**: Implement proper logging for debugging
6. **Testing**: Test integrations thoroughly
7. **Documentation**: Keep API usage documented

## Support

For API support and questions:
- Check the troubleshooting guide
- Review example integrations
- Consult the developer documentation
- Open issues on GitHub for bugs or feature requests