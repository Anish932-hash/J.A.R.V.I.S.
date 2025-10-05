# J.A.R.V.I.S. GUI - Advanced Graphical Interface

## Overview

J.A.R.V.I.S. features a cutting-edge graphical user interface built with PyQt6, offering both basic holographic and advanced 3D visualization modes. The GUI provides real-time system monitoring, voice control integration, and comprehensive control over all JARVIS subsystems.

## Features

### Core Features
- **Real-time System Monitoring**: Live CPU, memory, disk, and network monitoring
- **Voice Control Integration**: Seamless voice command interface
- **Command Console**: Interactive command execution with history
- **Multi-tab Interface**: Organized access to all JARVIS features
- **Futuristic Design**: Holographic effects and cyberpunk theming

### Advanced Features
- **3D System Visualization**: OpenGL-powered 3D charts and monitoring
- **Voice Waveform Display**: Real-time voice input visualization
- **API Status Monitoring**: Live tracking of AI provider connections
- **Self-Development Controls**: Direct access to JARVIS evolution features
- **Plugin Management**: Install and manage JARVIS extensions

## Installation Requirements

### Basic GUI
```bash
pip install PyQt6
```

### Advanced GUI (3D Visualizations)
```bash
pip install PyQt6 PyOpenGL PyQt6-Qt6
```

### Charts Support (Optional)
```bash
pip install PyQt6-Charts
```

## Usage

### GUI Launcher

The easiest way to start the GUI is using the launcher:

```bash
# Basic holographic GUI
python gui_launcher.py --basic

# Advanced GUI with 3D visualizations
python gui_launcher.py --advanced

# Auto-detect best available GUI
python gui_launcher.py
```

### Direct Launch

You can also launch GUI components directly:

```python
from gui.main_window import JARVISGUI
from jarvis import JARVIS

# Initialize JARVIS
jarvis = JARVIS()

# Launch GUI
gui = JARVISGUI(jarvis)
gui.show()
```

## Interface Overview

### Dashboard Tab
- **System Metrics Cards**: Real-time CPU, memory, disk, and network usage
- **Live Charts**: Performance monitoring with optional 3D visualization
- **Status Indicators**: System health and active module display
- **Real-time Clock**: Current time with live updates

### Command Interface Tab
- **Command Input**: Text-based command execution
- **Voice Control**: Toggle voice listening mode
- **Command History**: Browse and reuse previous commands
- **Output Console**: Formatted command results and system messages

### System Monitoring Tab
- **Process Monitor**: Real-time process listing with CPU/memory usage
- **Network Monitor**: Active network connections and interfaces
- **Performance Metrics**: Detailed system performance data

### Advanced Features Tab
- **Self-Development Engine**: Control JARVIS learning and evolution
- **Application Healer**: System health monitoring and auto-repair
- **IoT Integration**: Connected device management
- **API Status**: AI provider connection monitoring

### Security Tab
- **Threat Monitoring**: Real-time security event tracking
- **Access Control**: User authentication and permissions
- **Security Scanning**: Automated vulnerability assessment

### Plugin Management Tab
- **Installed Plugins**: Manage active JARVIS extensions
- **Plugin Marketplace**: Browse and install new plugins
- **Plugin Controls**: Load, unload, and configure plugins

## Voice Integration

### Voice Commands
- **Activation**: Say "JARVIS" to activate voice mode
- **Commands**: Natural language command execution
- **Feedback**: Audio and visual confirmation of voice commands

### Voice Settings
- **Microphone Calibration**: Optimize audio input quality
- **Voice Visualization**: Real-time waveform display during speech
- **Voice Selection**: Choose from available TTS voices

## Customization

### Themes
The GUI supports multiple visual themes:
- **Futuristic Dark**: Default cyberpunk theme with cyan accents
- **Cyberpunk**: High-contrast neon color scheme
- **Matrix**: Green monochrome terminal style
- **Neon Blue**: Electric blue holographic effects
- **Plasma**: Dynamic color-shifting theme

### Configuration
GUI settings can be customized in the JARVIS configuration file:

```json
{
  "gui": {
    "theme": "futuristic_dark",
    "animations_enabled": true,
    "holographic_effects": true,
    "voice_enabled": true,
    "chart_update_interval": 1000
  }
}
```

## Advanced Features

### 3D Visualization System
- **OpenGL Rendering**: Hardware-accelerated 3D graphics
- **Real-time Data**: Live system metrics in 3D space
- **Interactive Controls**: Rotate, zoom, and pan 3D views
- **Performance Optimized**: Efficient rendering for smooth animations

### API Integration Dashboard
- **Provider Status**: Real-time connection status for all AI providers
- **Usage Statistics**: API call counts and success rates
- **Error Monitoring**: Failed request tracking and diagnostics
- **Load Balancing**: Automatic failover between providers

### Self-Development Interface
- **Task Creation**: Initiate new development tasks
- **Progress Tracking**: Monitor ongoing improvements
- **Code Analysis**: Review generated and modified code
- **Performance Metrics**: Track improvement effectiveness

## Troubleshooting

### Common Issues

#### PyQt6 Not Available
```
Error: No module named 'PyQt6'
Solution: pip install PyQt6
```

#### OpenGL Not Working
```
Error: OpenGL context creation failed
Solution: Install graphics drivers or use --basic mode
```

#### Charts Not Displaying
```
Warning: Charts not available
Solution: pip install PyQt6-Charts
```

#### Voice Not Working
```
Error: Audio device not found
Solution: Check microphone permissions and drivers
```

### Performance Optimization

#### For Low-End Systems
- Use `--basic` mode to disable 3D visualizations
- Disable animations in settings
- Reduce chart update frequency

#### For High-End Systems
- Enable OpenGL mode for 3D visualizations
- Increase animation frame rate
- Enable advanced particle effects

## Development

### GUI Architecture

The GUI is built with a modular architecture:

```
gui/
├── __init__.py          # Main exports
├── main_window.py       # Basic holographic GUI
├── advanced_gui.py      # 3D visualization GUI
├── gui_launcher.py      # Cross-platform launcher
└── GUI_README.md       # This documentation
```

### Extending the GUI

#### Adding New Tabs
```python
def _create_custom_tab(self):
    """Create a custom tab"""
    tab = QWidget()
    layout = QVBoxLayout(tab)

    # Add your custom widgets here

    self.tab_widget.addTab(tab, "Custom Tab")
    return tab
```

#### Custom Themes
```python
def apply_custom_theme(self):
    """Apply custom theme"""
    self.setStyleSheet("""
        /* Custom styles here */
    """)
```

## API Reference

### JARVISGUI Class
Main GUI controller with system integration.

**Methods:**
- `show()`: Display the GUI
- `execute_command(command)`: Execute JARVIS command
- `update_status()`: Refresh system status
- `toggle_voice()`: Enable/disable voice control

### AdvancedJARVISGUI Class
Extended GUI with 3D capabilities.

**Methods:**
- `initialize_gui()`: Setup advanced interface
- `update_metrics()`: Refresh 3D visualizations
- `create_dev_task(type)`: Initiate development tasks

## License

This GUI is part of the J.A.R.V.I.S. system and follows the same licensing terms.

## Support

For GUI-related issues:
1. Check the troubleshooting section above
2. Verify PyQt6 installation: `python -c "import PyQt6; print('OK')"`
3. Test with basic mode: `python gui_launcher.py --basic`
4. Check logs in `jarvis_gui.log`

For feature requests or bug reports, please use the main JARVIS issue tracker.