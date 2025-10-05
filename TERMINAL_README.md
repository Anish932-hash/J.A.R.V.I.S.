# J.A.R.V.I.S. Terminal Mode

## Overview

J.A.R.V.I.S. Terminal Mode provides a beautiful, interactive terminal interface using the Rich library. This mode offers full functionality without requiring GUI libraries, making it perfect for servers, headless systems, or users who prefer terminal interfaces.

## Features

### Core Features
- **Rich Text Interface**: Beautiful, colorful terminal UI with tables, panels, and animations
- **Interactive API Setup**: Guided wizard for configuring AI provider API keys
- **Auto-Provider Selection**: Intelligent selection of best AI provider and model
- **Real-time Status**: Live monitoring of API providers and system status
- **Command History**: Persistent command history with navigation
- **Multi-panel Interface**: Organized dashboard with sidebar navigation

### AI Integration
- **100+ AI Providers**: Support for OpenAI, Anthropic, Google Gemini, Cohere, and more
- **Smart Load Balancing**: Automatic failover and provider optimization
- **Cost Tracking**: Real-time API usage and cost monitoring
- **Performance Analytics**: Response times, success rates, and usage statistics

## Installation

### Requirements
```bash
pip install rich aiohttp
```

### Optional Dependencies
```bash
# For enhanced features
pip install openai anthropic google-generativeai cohere replicate stability-sdk elevenlabs
```

## Usage

### Launch Terminal Mode

#### Using Main Launcher
```bash
python run.py --terminal
```

#### Direct Terminal Launcher
```bash
python run_terminal.py
```

#### Manual Launch
```bash
python -c "from terminal_interface import main; main()"
```

## First-Time Setup

When you first run terminal mode, you'll be guided through the API setup wizard:

1. **Welcome Screen**: Introduction to terminal mode
2. **Provider Overview**: Information about available AI providers
3. **Configuration Options**:
   - Configure individual providers
   - Auto-configure recommended providers
   - Skip and configure later
4. **API Key Input**: Secure password-masked input for API keys
5. **Validation**: Test connections to configured providers

## Interface Overview

### Main Layout
```
┌─ J.A.R.V.I.S. Terminal Interface | 14:30:25 ─┐
│                                                │
│ ┌─ Navigation ─┐ ┌─ Dashboard ─┐               │
│ │ 📊 Dashboard │ │ Welcome to  │               │
│ │ 🤖 AI Chat   │ │ J.A.R.V.I.S │               │
│ │ 🔧 Tools     │ │ Terminal... │               │
│ │ ⚙️ Settings  │ │             │               │
│ │ 📈 Analytics │ └─────────────┘               │
│ │ ❌ Exit      │                               │
│ └─────────────┘ ┌─ API Status ─┐               │
│ ┌─ System ─┐   │ Provider      │               │
│ │ API Prov │   │ Status        │               │
│ │ 5        │   │ Requests      │               │
│ │ Active   │   │ ...           │               │
│ │ Commands │   └───────────────┘               │
│ │ 42       │                                   │
│ └──────────┘                                   │
└────────────────────────────────────────────────┘
JARVIS> _
```

### Navigation Panels

#### Dashboard
- System welcome message
- API provider status overview
- Quick statistics

#### AI Chat
- Interactive AI conversation
- Command history display
- Voice status indicator

#### Tools
- API testing utilities
- Cost calculators
- System diagnostics

#### Settings
- Configuration management
- Theme selection
- Performance settings

#### Analytics
- API usage statistics
- Cost tracking
- Performance metrics

## Commands

### Navigation Commands
- `dashboard, d` - Show main dashboard
- `chat, c` - Enter AI chat mode
- `tools, t` - Show available tools
- `settings, s` - Show settings panel
- `analytics, a` - Show API analytics
- `back` - Return to dashboard

### AI Commands
- `<any text>` - Send message to AI for processing
- `api status` - Show detailed API provider status
- `api test <provider>` - Test specific provider connectivity
- `api switch <provider>` - Switch active provider
- `api help` - Show API command help

### System Commands
- `help` - Show all available commands
- `exit, quit, q` - Exit terminal mode

## API Provider Management

### Supported Providers

| Provider | Type | Best For | Setup Required |
|----------|------|----------|----------------|
| OpenAI | Text | General AI, Coding | API Key |
| Anthropic | Text | Analysis, Reasoning | API Key |
| Google Gemini | Text/Vision | Multimodal, Fast | API Key |
| Cohere | Text | Creative Writing | API Key |
| Stability AI | Image | Image Generation | API Key |
| ElevenLabs | Audio | Text-to-Speech | API Key |

### Auto-Selection Algorithm

The terminal mode automatically selects the best provider and model based on:

1. **Provider Priority**: OpenAI > Anthropic > Google > Others
2. **Recent Performance**: Providers used successfully recently
3. **Cost Efficiency**: Lower cost per token preferred
4. **Success Rate**: Higher success rate prioritized
5. **Response Time**: Faster providers preferred

### Configuration Storage

API keys are stored securely in:
- `config/api_config.json` - Persistent configuration
- Environment variables (temporary session)

## Advanced Features

### Smart Command Processing

Terminal mode includes intelligent command processing:

- **Natural Language**: Understands conversational commands
- **Context Awareness**: Maintains conversation context
- **Multi-turn Dialogues**: Supports complex interactions
- **Error Recovery**: Automatic retry on failures

### Real-time Monitoring

- **Provider Health**: Continuous monitoring of API endpoints
- **Usage Tracking**: Real-time cost and usage monitoring
- **Performance Metrics**: Response times and success rates
- **System Resources**: Local system monitoring

### Cost Management

- **Budget Tracking**: Monitor API usage costs
- **Cost Alerts**: Warnings when approaching limits
- **Usage Optimization**: Automatic provider switching for cost efficiency
- **Detailed Reporting**: Cost breakdown by provider and model

## Customization

### Themes
Terminal mode supports multiple visual themes:
- **Default**: Blue/cyan color scheme
- **Dark**: High contrast dark theme
- **Minimal**: Clean, minimal interface
- **Colorful**: Vibrant color scheme

### Configuration
Settings can be modified in the interface or config files:

```json
{
  "terminal": {
    "theme": "default",
    "max_history": 100,
    "auto_save": true,
    "cost_alerts": true,
    "budget_limit": 10.0
  }
}
```

## Troubleshooting

### Common Issues

#### Rich Library Not Found
```
Error: No module named 'rich'
Solution: pip install rich
```

#### API Connection Failed
```
Error: Provider connection failed
Solution:
1. Check API key validity
2. Verify internet connection
3. Check provider status: api status
4. Test specific provider: api test <provider>
```

#### No Providers Configured
```
Warning: No AI providers configured
Solution:
1. Run setup wizard: python run_terminal.py
2. Configure manually: api setup
3. Check environment variables
```

#### High Costs
```
Warning: API costs exceeding budget
Solution:
1. Check usage: analytics
2. Switch providers: api switch <cheaper_provider>
3. Set budget limits in settings
```

### Performance Optimization

#### For Slow Systems
- Reduce animation frequency
- Disable real-time updates
- Use simpler themes

#### For High Usage
- Set budget limits
- Enable cost alerts
- Monitor usage regularly

## Development

### Architecture

```
terminal_interface.py
├── TerminalInterface (main class)
│   ├── Rich console and UI components
│   ├── API manager integration
│   ├── Command processing
│   └── Panel management
├── API Setup Wizard
│   ├── Provider discovery
│   ├── Key validation
│   └── Configuration saving
└── Auto-selection Engine
    ├── Provider scoring
    ├── Model selection
    └── Performance tracking
```

### Extending Terminal Mode

#### Adding New Panels
```python
def create_custom_panel(self) -> Panel:
    """Create a custom panel"""
    content = "Custom panel content"
    return Panel(content, title="Custom Panel")
```

#### Adding Commands
```python
def handle_custom_command(self, command: str):
    """Handle custom commands"""
    if command.startswith("custom"):
        # Process custom command
        pass
```

#### Custom Themes
```python
def apply_theme(self, theme_name: str):
    """Apply custom theme"""
    themes = {
        "custom": {
            "header": "bold magenta",
            "success": "green",
            "error": "red"
        }
    }
    self.styles.update(themes.get(theme_name, {}))
```

## API Reference

### TerminalInterface Class

#### Methods
- `run()` - Start terminal interface
- `show_help()` - Display help information
- `handle_user_input()` - Process user commands
- `auto_select_provider_and_model()` - Smart provider selection

#### Properties
- `console` - Rich console instance
- `api_manager` - API manager instance
- `current_panel` - Current active panel
- `command_history` - Command history list

## Security

### API Key Protection
- Password-masked input for API keys
- Secure storage in configuration files
- No logging of sensitive information
- Environment variable support for CI/CD

### Data Privacy
- Local processing of all data
- No telemetry or data collection
- Secure API communication
- Configurable data retention

## License

Terminal mode is part of J.A.R.V.I.S. and follows the main project licensing.

## Support

For terminal mode issues:
1. Check this documentation
2. Verify Rich installation: `python -c "import rich; print('OK')"`
3. Test basic functionality: `python run_terminal.py`
4. Check logs for error details
5. Report issues on the main J.A.R.V.I.S. repository

---

**Pro Tip**: Use `help` command in terminal mode for interactive assistance, or `api help` for API-specific commands.