#!/usr/bin/env python3
"""
J.A.R.V.I.S. Terminal Interface
Advanced terminal-based GUI using Rich library for beautiful text interfaces
"""

import os
import sys
import time
import asyncio
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

# Rich imports for beautiful terminal interface
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.columns import Columns
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.status import Status
    from rich.align import Align
    from rich.style import Style
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Add JARVIS to path
sys.path.insert(0, os.path.dirname(__file__))

from core.jarvis import JARVIS
from core.command_processor import CommandProcessor
from core.api_manager import APIManager, APIProvider, APIRequest, APIResponse


class TerminalInterface:
    """
    Advanced terminal interface for J.A.R.V.I.S. using Rich library
    Provides beautiful text-based GUI with real-time updates
    """

    def __init__(self):
        if not RICH_AVAILABLE:
            print("ERROR: Rich library not available. Install with: pip install rich")
            sys.exit(1)

        # Initialize console with safe encoding to avoid Unicode issues
        self.console = Console(safe_box=True, legacy_windows=False)
        self.api_manager = None
        self.jarvis_instance = None
        self.current_mode = "setup"
        self.system_status = {}
        self.command_history = []
        self.max_history = 100

        # Interface state
        self.running = False
        self.current_panel = "dashboard"

        # Colors and styles
        self.styles = {
            'header': Style(color="cyan", bold=True),
            'success': Style(color="green", bold=True),
            'error': Style(color="red", bold=True),
            'warning': Style(color="yellow", bold=True),
            'info': Style(color="blue", bold=True),
            'accent': Style(color="magenta", bold=True),
            'dim': Style(color="white", dim=True)
        }

    def run(self):
        """Main interface loop"""
        try:
            self.console.clear()
            self.show_welcome()

            # Initialize JARVIS
            if not self.initialize_jarvis():
                return

            self.running = True
            self.main_loop()

        except KeyboardInterrupt:
            self.console.print("\n[bold cyan]👋 Goodbye![/bold cyan]")
        except Exception as e:
            self.console.print(f"[bold red]❌ Fatal error: {e}[/bold red]")
        finally:
            self.cleanup()

    def show_welcome(self):
        """Show welcome screen"""
        welcome_text = Text("J.A.R.V.I.S. Terminal Interface", style="bold cyan")
        welcome_panel = Panel(
            Align.center(welcome_text),
            title="[bold blue]Welcome[/bold blue]",
            border_style="blue"
        )
        self.console.print(welcome_panel)
        self.console.print()

    async def initialize_jarvis_async(self) -> bool:
        """Initialize JARVIS and API manager with optimized loading for terminal mode"""
        try:
            with self.console.status("[bold green]Initializing JARVIS Terminal Mode...", spinner="dots"):
                # Create full JARVIS instance but disable heavy components for terminal mode
                self.jarvis_instance = JARVIS()

                # Disable heavy AI components that slow down terminal startup
                if hasattr(self.jarvis_instance, 'config') and 'system' in self.jarvis_instance.config:
                    # Keep voice interface for potential future use but skip heavy AI components
                    self.jarvis_instance.config['system']['enable_voice'] = False  # Skip voice for faster startup

                # Initialize only essential JARVIS modules (skip heavy AI components)
                await self._initialize_essential_modules_only()

                # Create separate API manager for AI chat functionality
                self.api_manager = APIManager(self.jarvis_instance)

                # Run API setup wizard
                if not self.run_api_setup_wizard():
                    return False

                # Initialize API manager
                await self.api_manager.initialize()

            self.console.print("[bold green]✅ JARVIS Terminal Mode initialized successfully![/bold green]")
            time.sleep(1)
            return True

        except Exception as e:
            self.console.print(f"[bold red]❌ Failed to initialize JARVIS: {e}[/bold red]")
            return False

    def initialize_jarvis(self) -> bool:
        """Wrapper to initialize JARVIS asynchronously"""
        return asyncio.run(self.initialize_jarvis_async())

    async def _initialize_essential_modules_only(self):
        """Initialize only essential modules for terminal mode, skip heavy AI components"""
        try:
            self.jarvis_instance.logger.info("Initializing essential modules for terminal mode...")

            # Initialize only core system modules needed for terminal functionality

            # System monitor (lightweight, needed for basic system info)
            from jarvis.modules.system_monitor import SystemMonitor
            self.jarvis_instance.system_monitor = SystemMonitor(self.jarvis_instance)
            self.jarvis_instance.system_monitor.start_monitoring()

            # Application controller (essential for opening applications)
            from jarvis.modules.application_controller import ApplicationController
            self.jarvis_instance.app_controller = ApplicationController(self.jarvis_instance)
            self.jarvis_instance.app_controller.initialize()

            # File manager (essential for file operations)
            from jarvis.modules.file_manager import FileManager
            self.jarvis_instance.file_manager = FileManager(self.jarvis_instance)
            self.jarvis_instance.file_manager.initialize()

            # Command processor (essential for command execution)
            self.jarvis_instance.command_processor = CommandProcessor(self.jarvis_instance)

            # Skip all heavy AI components that cause slow startup:
            # - Voice interface (loads TTS engines)
            # - Network manager (scans network interfaces)
            # - Security manager (loads encryption)
            # - Plugin manager (loads plugins)
            # - All advanced AI components (SelfDevelopmentEngine, ApplicationHealer, etc.)

            self.jarvis_instance.logger.info("Essential modules initialized for terminal mode")

        except Exception as e:
            self.jarvis_instance.logger.error(f"Error initializing essential modules: {str(e)}")
            raise

    def run_api_setup_wizard(self) -> bool:
        """Run interactive API setup wizard"""
        self.console.clear()
        title = Text("API Configuration Wizard", style="bold cyan")
        setup_panel = Panel(
            Align.center(title),
            title="[bold yellow]Setup[/bold yellow]",
            border_style="yellow"
        )
        self.console.print(setup_panel)
        self.console.print()

        # Show available providers
        self.show_available_providers()

        # Ask user to configure providers
        configured_providers = []

        while True:
            self.console.print("\n[bold cyan]Available API Providers:[/bold cyan]")
            providers_table = Table(show_header=True, header_style="bold magenta")
            providers_table.add_column("Provider", style="cyan")
            providers_table.add_column("Type", style="green")
            providers_table.add_column("Status", style="yellow")

            for provider in APIProvider:
                status = "Not Configured"
                if hasattr(self.api_manager, 'api_configs') and provider in self.api_manager.api_configs:
                    config = self.api_manager.api_configs[provider]
                    status = "Configured" if config.api_key else "No API Key"

                provider_type = "Text"
                if provider in [APIProvider.STABILITY_AI, APIProvider.DALL_E, APIProvider.MIDJOURNEY]:
                    provider_type = "Image"
                elif provider in [APIProvider.ELEVENLABS, APIProvider.RESPEECH]:
                    provider_type = "Audio"

                providers_table.add_row(provider.value.replace('_', ' ').title(), provider_type, status)

            self.console.print(providers_table)

            # Ask user what to do
            self.console.print("\n[bold white]Options:[/bold white]")
            self.console.print("1. Configure a provider")
            self.console.print("2. Auto-configure recommended providers")
            self.console.print("3. Continue with current configuration")
            self.console.print("4. Exit setup")

            choice = Prompt.ask("\nEnter your choice", choices=["1", "2", "3", "4"], default="2")

            if choice == "1":
                if self.configure_single_provider():
                    configured_providers.append("manual")
            elif choice == "2":
                if self.auto_configure_providers():
                    configured_providers.append("auto")
                    break
            elif choice == "3":
                if self.check_minimum_configuration():
                    break
                else:
                    self.console.print("[bold yellow]⚠️  Minimum configuration not met. Please configure at least one text provider.[/bold yellow]")
            elif choice == "4":
                return False

        self.console.print(f"\n[bold green]✅ API setup completed! Configured providers: {len(configured_providers)}[/bold green]")
        return True

    def show_available_providers(self):
        """Show information about available providers"""
        info_table = Table(title="Available AI Providers", show_header=True, header_style="bold blue")
        info_table.add_column("Category", style="cyan", no_wrap=True)
        info_table.add_column("Providers", style="white")
        info_table.add_column("Best For", style="green")

        info_table.add_row(
            "Text Generation",
            "OpenAI, Anthropic, Google Gemini, Cohere",
            "General AI conversations, coding, analysis"
        )
        info_table.add_row(
            "Image Generation",
            "Stability AI, Midjourney, DALL-E",
            "Creating images from text descriptions"
        )
        info_table.add_row(
            "Audio/Speech",
            "ElevenLabs, Respeecher",
            "Text-to-speech, voice synthesis"
        )
        info_table.add_row(
            "Code Generation",
            "GitHub Copilot, Tabnine",
            "Programming assistance, code completion"
        )

        self.console.print(info_table)
        self.console.print()

    def configure_single_provider(self) -> bool:
        """Configure a single provider interactively"""
        try:
            # Show provider list
            providers = [p.value for p in APIProvider]
            provider_display = [p.replace('_', ' ').title() for p in providers]

            self.console.print("\n[bold cyan]Select provider to configure:[/bold cyan]")
            for i, name in enumerate(provider_display, 1):
                self.console.print(f"{i}. {name}")

            choice = IntPrompt.ask("Enter provider number", min_=1, max_=len(providers))
            selected_provider = list(APIProvider)[choice - 1]

            # Get API key
            provider_name = selected_provider.value.replace('_', ' ').title()
            api_key = Prompt.ask(f"Enter {provider_name} API key", password=True)

            if not api_key.strip():
                self.console.print("[bold yellow]⚠️  No API key entered. Skipping configuration.[/bold yellow]")
                return False

            # Set environment variable
            env_var = f"{selected_provider.value.upper()}_API_KEY"
            os.environ[env_var] = api_key

            # Update API manager config
            if hasattr(self.api_manager, '_load_api_configurations'):
                # Reinitialize configurations
                asyncio.run(self.api_manager._load_api_configurations())

            # Save configuration
            if hasattr(self.api_manager, 'save_api_configurations'):
                self.api_manager.save_api_configurations()

            self.console.print(f"[bold green]✅ {provider_name} configured successfully![/bold green]")
            return True

        except Exception as e:
            self.console.print(f"[bold red]❌ Error configuring provider: {e}[/bold red]")
            return False

    def auto_configure_providers(self) -> bool:
        """Auto-configure recommended providers"""
        try:
            self.console.print("\n[bold cyan]🔧 Auto-configuring recommended providers...[/bold cyan]")

            # Define priority providers
            priority_providers = [
                (APIProvider.OPENAI, "GPT-4"),
                (APIProvider.ANTHROPIC, "Claude-3"),
                (APIProvider.GOOGLE_GEMINI, "Gemini Pro"),
                (APIProvider.STABILITY_AI, "Stable Diffusion"),
                (APIProvider.ELEVENLABS, "ElevenLabs TTS")
            ]

            configured_count = 0

            for provider, model_name in priority_providers:
                provider_name = provider.value.replace('_', ' ').title()

                # Check if already configured
                env_var = f"{provider.value.upper()}_API_KEY"
                if os.getenv(env_var):
                    self.console.print(f"[dim]⏭️  {provider_name} already configured[/dim]")
                    configured_count += 1
                    continue

                # Ask user if they want to configure this provider
                if Confirm.ask(f"Configure {provider_name} ({model_name})?", default=False):
                    api_key = Prompt.ask(f"Enter {provider_name} API key", password=True)

                    if api_key.strip():
                        os.environ[env_var] = api_key
                        configured_count += 1
                        self.console.print(f"[bold green]✅ {provider_name} configured![/bold green]")

                        # Save configuration
                        if hasattr(self.api_manager, 'save_api_configurations'):
                            self.api_manager.save_api_configurations()
                    else:
                        self.console.print(f"[dim]⏭️  Skipped {provider_name}[/dim]")
                else:
                    self.console.print(f"[dim]⏭️  Skipped {provider_name}[/dim]")

            if configured_count > 0:
                # Reinitialize configurations
                if hasattr(self.api_manager, '_load_api_configurations'):
                    asyncio.run(self.api_manager._load_api_configurations())

                self.console.print(f"\n[bold green]✅ Auto-configuration completed! {configured_count} providers configured.[/bold green]")
                return True
            else:
                self.console.print("[bold yellow]⚠️  No providers configured. Please configure at least one text provider.[/bold yellow]")
                return False

        except Exception as e:
            self.console.print(f"[bold red]❌ Error in auto-configuration: {e}[/bold red]")
            return False

    def check_minimum_configuration(self) -> bool:
        """Check if minimum configuration is met"""
        if not hasattr(self.api_manager, 'api_configs'):
            return False

        # Check for at least one text provider
        text_providers = [APIProvider.OPENAI, APIProvider.ANTHROPIC, APIProvider.GOOGLE_GEMINI, APIProvider.COHERE]

        for provider in text_providers:
            if provider in self.api_manager.api_configs:
                config = self.api_manager.api_configs[provider]
                if config.api_key and config.enabled:
                    return True

        return False

    def main_loop(self):
        """Main interface loop"""
        try:
            while self.running:
                self.console.clear()
                self.show_header()
                self.show_main_interface()

                # Handle user input
                if not self.handle_user_input():
                    break

        except Exception as e:
            self.console.print(f"[bold red]❌ Error in main loop: {e}[/bold red]")

    def show_header(self):
        """Show interface header"""
        current_time = datetime.now().strftime("%H:%M:%S")
        header_text = f"J.A.R.V.I.S. Terminal Interface | {current_time}"

        header_panel = Panel(
            Align.center(Text(header_text, style="bold cyan")),
            border_style="cyan",
            padding=(0, 1)
        )
        self.console.print(header_panel)

    def show_main_interface(self):
        """Show main interface with panels"""
        # Create layout
        layout = Layout()

        # Split into sections
        layout.split_row(
            Layout(name="sidebar", size=30),
            Layout(name="main", ratio=2)
        )

        # Sidebar with navigation
        layout["sidebar"].update(self.create_sidebar())

        # Main content area
        layout["main"].update(self.create_main_content())

        self.console.print(layout)

    def create_sidebar(self) -> Panel:
        """Create sidebar with navigation and status"""
        sidebar_content = []

        # Navigation menu
        nav_table = Table(show_header=False, box=None)
        nav_table.add_column("Menu", style="cyan")

        menu_items = [
            ("[D] Dashboard", "dashboard"),
            ("[C] AI Chat", "chat"),
            ("[T] Tools", "tools"),
            ("[S] Settings", "settings"),
            ("[A] Analytics", "analytics"),
            ("[X] Exit", "exit")
        ]

        for item, panel in menu_items:
            style = "bold cyan" if panel == self.current_panel else "white"
            nav_table.add_row(f"[{style}]{item}[/{style}]")

        sidebar_content.append(nav_table)
        sidebar_content.append("")  # Spacer

        # System status
        status_table = Table(title="System Status", show_header=False)
        status_table.add_column("Metric", style="cyan")
        status_table.add_column("Value", style="green")

        status_items = [
            ("API Providers", f"{len(self.api_manager.api_configs) if self.api_manager else 0}"),
            ("Active", f"{len([c for c in self.api_manager.api_configs.values() if c.enabled]) if self.api_manager else 0}"),
            ("Commands", f"{len(self.command_history)}"),
            ("Uptime", "00:00:00")  # Would track actual uptime
        ]

        for metric, value in status_items:
            status_table.add_row(metric, value)

        sidebar_content.append(status_table)

        return Panel(
            "\n".join(str(content) for content in sidebar_content),
            title="[bold blue]Navigation[/bold blue]",
            border_style="blue"
        )

    def create_main_content(self) -> Panel:
        """Create main content area"""
        if self.current_panel == "dashboard":
            return self.create_dashboard_panel()
        elif self.current_panel == "chat":
            return self.create_chat_panel()
        elif self.current_panel == "tools":
            return self.create_tools_panel()
        elif self.current_panel == "settings":
            return self.create_settings_panel()
        elif self.current_panel == "analytics":
            return self.create_analytics_panel()
        else:
            return Panel("Unknown panel", title="Error")

    def create_dashboard_panel(self) -> Panel:
        """Create dashboard panel"""
        content = []

        # Welcome message
        welcome = Panel(
            Align.center("Welcome to J.A.R.V.I.S. Terminal Interface\nType 'help' for commands"),
            border_style="green"
        )
        content.append(welcome)
        content.append("")

        # API Status
        if self.api_manager:
            api_table = Table(title="API Provider Status", show_header=True)
            api_table.add_column("Provider", style="cyan")
            api_table.add_column("Status", style="green")
            api_table.add_column("Requests", style="yellow", justify="right")

            for provider, config in self.api_manager.api_configs.items():
                status = "✅ Active" if config.enabled else "❌ Disabled"
                requests = str(config.total_requests)
                api_table.add_row(provider.value.replace('_', ' ').title(), status, requests)

            content.append(api_table)

        return Panel(
            "\n".join(str(item) for item in content),
            title="[bold cyan]Dashboard[/bold cyan]",
            border_style="cyan"
        )

    def create_chat_panel(self) -> Panel:
        """Create AI chat panel"""
        chat_content = []

        # Chat history (last few messages)
        if self.command_history:
            history_table = Table(title="Recent Commands", show_header=False)
            history_table.add_column("Command", style="cyan")

            for cmd in self.command_history[-5:]:
                history_table.add_row(cmd[:50] + "..." if len(cmd) > 50 else cmd)

            chat_content.append(history_table)
            chat_content.append("")

        # Input prompt
        chat_content.append("[bold green]Enter your message or command:[/bold green]")
        chat_content.append("[dim](Type 'back' to return to dashboard)[/dim]")

        return Panel(
            "\n".join(chat_content),
            title="[bold magenta]AI Chat[/bold magenta]",
            border_style="magenta"
        )

    def create_tools_panel(self) -> Panel:
        """Create tools panel"""
        tools_content = []

        tools_table = Table(title="Available Tools", show_header=True)
        tools_table.add_column("Tool", style="cyan")
        tools_table.add_column("Description", style="white")
        tools_table.add_column("Status", style="green")

        tools = [
            ("API Test", "Test API provider connectivity", "Available"),
            ("Cost Calculator", "Calculate API usage costs", "Available"),
            ("Performance Monitor", "Monitor system performance", "Available"),
            ("Configuration Editor", "Edit JARVIS settings", "Available")
        ]

        for tool, desc, status in tools:
            tools_table.add_row(tool, desc, status)

        tools_content.append(tools_table)

        return Panel(
            "\n".join(str(item) for item in tools_content),
            title="[bold yellow]Tools[/bold yellow]",
            border_style="yellow"
        )

    def create_settings_panel(self) -> Panel:
        """Create settings panel"""
        settings_content = []

        settings_table = Table(title="Settings", show_header=True)
        settings_table.add_column("Setting", style="cyan")
        settings_table.add_column("Value", style="white")
        settings_table.add_column("Description", style="dim")

        settings = [
            ("Theme", "Default", "Interface color scheme"),
            ("Auto-save", "Enabled", "Automatically save configuration"),
            ("Command History", f"{self.max_history}", "Maximum commands to remember"),
            ("API Timeout", "30s", "API request timeout"),
            ("Debug Mode", "Disabled", "Enable debug logging")
        ]

        for setting, value, desc in settings:
            settings_table.add_row(setting, value, desc)

        settings_content.append(settings_table)

        return Panel(
            "\n".join(str(item) for item in settings_content),
            title="[bold red]Settings[/bold red]",
            border_style="red"
        )

    def create_analytics_panel(self) -> Panel:
        """Create analytics panel"""
        analytics_content = []

        if self.api_manager:
            stats = self.api_manager.get_stats()

            analytics_table = Table(title="API Analytics", show_header=True)
            analytics_table.add_column("Metric", style="cyan")
            analytics_table.add_column("Value", style="green", justify="right")

            analytics_data = [
                ("Total Requests", str(stats.get('total_requests', 0))),
                ("Successful", str(stats.get('successful_requests', 0))),
                ("Failed", str(stats.get('failed_requests', 0))),
                ("Cache Hits", str(stats.get('cache_hits', 0))),
                ("Total Cost", f"${stats.get('total_cost', 0):.2f}"),
                ("Avg Response Time", f"{stats.get('average_response_time', 0):.2f}s")
            ]

            for metric, value in analytics_data:
                analytics_table.add_row(metric, value)

            analytics_content.append(analytics_table)

        return Panel(
            "\n".join(str(item) for item in analytics_content) if analytics_content else "No analytics data available",
            title="[bold blue]Analytics[/bold blue]",
            border_style="blue"
        )

    def handle_user_input(self) -> bool:
        """Handle user input"""
        try:
            # Show command prompt
            self.console.print()
            command = Prompt.ask("[bold cyan]JARVIS>[/bold cyan]").strip()

            if not command:
                return True

            # Add to history
            self.command_history.append(command)
            if len(self.command_history) > self.max_history:
                self.command_history.pop(0)

            # Process command
            if command.lower() in ['exit', 'quit', 'q']:
                return False
            elif command.lower() == 'back':
                self.current_panel = "dashboard"
            elif command.lower() in ['dashboard', 'd']:
                self.current_panel = "dashboard"
            elif command.lower() in ['chat', 'c']:
                self.current_panel = "chat"
            elif command.lower() in ['tools', 't']:
                self.current_panel = "tools"
            elif command.lower() in ['settings', 's']:
                self.current_panel = "settings"
            elif command.lower() in ['analytics', 'a']:
                self.current_panel = "analytics"
            elif command.lower() == 'help':
                self.show_help()
                input("\nPress Enter to continue...")
            elif command.lower().startswith('api'):
                self.handle_api_command(command)
            else:
                # Process as AI command
                self.process_ai_command(command)

            return True

        except KeyboardInterrupt:
            return False
        except Exception as e:
            self.console.print(f"[bold red]❌ Error processing command: {e}[/bold red]")
            return True

    def show_help(self):
        """Show help information"""
        help_table = Table(title="J.A.R.V.I.S. Terminal Commands", show_header=True)
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description", style="white")
        help_table.add_column("Example", style="green")

        commands = [
            ("Navigation", "", ""),
            ("dashboard, d", "Show main dashboard", "dashboard"),
            ("chat, c", "Enter AI chat mode", "chat"),
            ("tools, t", "Show available tools", "tools"),
            ("settings, s", "Show settings", "settings"),
            ("analytics, a", "Show API analytics", "analytics"),
            ("back", "Return to dashboard", "back"),
            ("", "", ""),
            ("AI Commands", "", ""),
            ("<any text>", "Send to AI for processing", "Hello JARVIS"),
            ("", "", ""),
            ("API Commands", "", ""),
            ("api status", "Show API provider status", "api status"),
            ("api test <provider>", "Test API provider", "api test openai"),
            ("api switch <provider>", "Switch active provider", "api switch anthropic"),
            ("", "", ""),
            ("System", "", ""),
            ("help", "Show this help", "help"),
            ("exit, quit, q", "Exit JARVIS", "exit")
        ]

        for cmd, desc, example in commands:
            help_table.add_row(cmd, desc, example)

        self.console.print(help_table)

    def handle_api_command(self, command: str):
        """Handle API-related commands"""
        try:
            parts = command.split()
            if len(parts) < 2:
                self.console.print("[bold red]❌ Invalid API command. Use 'api help' for usage.[/bold red]")
                return

            subcommand = parts[1].lower()

            if subcommand == "status":
                self.show_api_status()
            elif subcommand == "test" and len(parts) >= 3:
                provider_name = parts[2].lower()
                self.test_api_provider(provider_name)
            elif subcommand == "switch" and len(parts) >= 3:
                provider_name = parts[2].lower()
                self.switch_api_provider(provider_name)
            elif subcommand == "help":
                self.show_api_help()
            else:
                self.console.print(f"[bold red]❌ Unknown API command: {subcommand}[/bold red]")

        except Exception as e:
            self.console.print(f"[bold red]❌ Error executing API command: {e}[/bold red]")

    def show_api_status(self):
        """Show API provider status"""
        if not self.api_manager:
            self.console.print("[bold red]❌ API manager not initialized[/bold red]")
            return

        status_table = Table(title="API Provider Status", show_header=True)
        status_table.add_column("Provider", style="cyan")
        status_table.add_column("Status", style="green")
        status_table.add_column("Requests", style="yellow", justify="right")
        status_table.add_column("Success Rate", style="blue", justify="right")
        status_table.add_column("Cost", style="magenta", justify="right")

        for provider, config in self.api_manager.api_configs.items():
            status = "[green]Active[/green]" if config.enabled else "[red]Disabled[/red]"
            requests = str(config.total_requests)
            success_rate = ".0%"
            if config.total_requests > 0:
                rate = (config.total_requests - config.failed_requests) / config.total_requests
                success_rate = f"{rate:.1%}"
            cost = f"${config.total_cost:.2f}"

            status_table.add_row(
                provider.value.replace('_', ' ').title(),
                status, requests, success_rate, cost
            )

        self.console.print(status_table)

    def test_api_provider(self, provider_name: str):
        """Test API provider connectivity"""
        try:
            # Find provider
            target_provider = None
            for provider in APIProvider:
                if provider.value.lower() == provider_name.lower():
                    target_provider = provider
                    break

            if not target_provider:
                self.console.print(f"[bold red]❌ Provider '{provider_name}' not found[/bold red]")
                return

            if target_provider not in self.api_manager.api_configs:
                self.console.print(f"[bold red]❌ Provider '{provider_name}' not configured[/bold red]")
                return

            # Test provider
            with self.console.status(f"[bold green]Testing {provider_name}...", spinner="dots"):
                config = self.api_manager.api_configs[target_provider]

                # Create test request
                test_request = APIRequest(
                    provider=target_provider,
                    model=config.models[0] if config.models else "test",
                    prompt="Hello, this is a test message.",
                    timeout=10
                )

                # Execute test
                response = asyncio.run(self.api_manager.make_request(test_request))

            if response.success:
                self.console.print(f"[bold green]✅ {provider_name} test successful![/bold green]")
                self.console.print(f"[dim]Response time: {response.response_time:.2f}s[/dim]")
                if response.cost > 0:
                    self.console.print(f"[dim]Cost: ${response.cost:.4f}[/dim]")
            else:
                self.console.print(f"[bold red]❌ {provider_name} test failed: {response.error}[/bold red]")

        except Exception as e:
            self.console.print(f"[bold red]❌ Error testing provider: {e}[/bold red]")

    def switch_api_provider(self, provider_name: str):
        """Switch active API provider"""
        try:
            # Find provider
            target_provider = None
            for provider in APIProvider:
                if provider.value.lower() == provider_name.lower():
                    target_provider = provider
                    break

            if not target_provider:
                self.console.print(f"[bold red]❌ Provider '{provider_name}' not found[/bold red]")
                return

            if target_provider not in self.api_manager.api_configs:
                self.console.print(f"[bold red]❌ Provider '{provider_name}' not configured[/bold red]")
                return

            # Enable target provider, disable others in same category
            config = self.api_manager.api_configs[target_provider]
            config.enabled = True

            self.console.print(f"[bold green]✅ Switched to {provider_name}[/bold green]")

        except Exception as e:
            self.console.print(f"[bold red]❌ Error switching provider: {e}[/bold red]")

    def show_api_help(self):
        """Show API command help"""
        api_help = Table(title="API Commands Help", show_header=True)
        api_help.add_column("Command", style="cyan")
        api_help.add_column("Description", style="white")
        api_help.add_column("Example", style="green")

        commands = [
            ("api status", "Show all API provider status", "api status"),
            ("api test <provider>", "Test provider connectivity", "api test openai"),
            ("api switch <provider>", "Switch active provider", "api switch anthropic"),
            ("api help", "Show this help", "api help")
        ]

        for cmd, desc, example in commands:
            api_help.add_row(cmd, desc, example)

        self.console.print(api_help)

    def process_ai_command(self, command: str):
        """Process natural language command with AI assistance"""
        try:
            # Use AI to understand and execute the user's request
            with self.console.status("[bold green]Processing your request...", spinner="dots"):
                # Auto-select best provider and model
                provider, model = self.auto_select_provider_and_model(command)

                if not provider:
                    self.console.print("[bold red]❌ No suitable AI provider available[/bold red]")
                    return

                # Create enhanced prompt that includes system capabilities
                enhanced_prompt = f"""
You are JARVIS, an advanced AI assistant with system control capabilities.

User request: {command}

You have access to these system functions:
- Open/close applications (e.g., "open notepad", "close chrome")
- System information (CPU, memory, disk usage)
- File operations (create, delete, search files)
- Time and date queries
- Volume control
- Web search and research
- Code analysis and generation

If the user is asking for a system action, respond with a brief confirmation and execute it.
If they want information or conversation, provide a helpful response.
If they want you to perform a task, do it and confirm completion.

Keep responses concise but helpful.
"""

                # Create API request
                request = APIRequest(
                    provider=provider,
                    model=model,
                    prompt=enhanced_prompt,
                    parameters={"temperature": 0.3, "max_tokens": 300}  # Lower temperature for more consistent responses
                )

                # Execute request
                response = asyncio.run(self.api_manager.make_request(request))

            if response.success:
                ai_response = response.response.strip()

                # Check if AI suggests a system command
                if self.jarvis_instance and self._should_execute_command(ai_response, command):
                    # Try to execute the suggested command
                    try:
                        system_result = self.jarvis_instance.execute_command(command)

                        if system_result and isinstance(system_result, dict) and system_result.get("success"):
                            # Show AI response and command result
                            response_panel = Panel(
                                f"{ai_response}\n\n✅ {system_result.get('message', 'Action completed')}",
                                title="[bold green]JARVIS Response[/bold green]",
                                border_style="green"
                            )
                            self.console.print(response_panel)
                        else:
                            # Show AI response only
                            response_panel = Panel(
                                ai_response,
                                title="[bold green]JARVIS Response[/bold green]",
                                border_style="green"
                            )
                            self.console.print(response_panel)

                    except Exception as cmd_error:
                        # Show AI response with note about command issue
                        response_panel = Panel(
                            f"{ai_response}\n\n[yellow]Note: Could not execute system command automatically[/yellow]",
                            title="[bold green]JARVIS Response[/bold green]",
                            border_style="green"
                        )
                        self.console.print(response_panel)
                else:
                    # Regular AI response
                    response_panel = Panel(
                        ai_response,
                        title="[bold green]JARVIS Response[/bold green]",
                        border_style="green"
                    )
                    self.console.print(response_panel)

                # Show metadata
                metadata = f"Response time: {response.response_time:.2f}s"
                if response.cost > 0:
                    metadata += f" | Cost: ${response.cost:.4f}"
                if response.tokens_used > 0:
                    metadata += f" | Tokens: {response.tokens_used}"

                self.console.print(f"[dim]{metadata}[/dim]")
            else:
                self.console.print(f"[bold red]❌ AI request failed: {response.error}[/bold red]")

        except Exception as e:
            self.console.print(f"[bold red]❌ Error processing request: {e}[/bold red]")

    def _should_execute_command(self, ai_response: str, original_command: str) -> bool:
        """Determine if the AI response suggests executing a system command"""
        # Check for command-like patterns in the original request
        command_indicators = [
            'open', 'launch', 'start', 'close', 'quit', 'exit',
            'create', 'delete', 'search', 'find',
            'volume', 'time', 'date', 'status'
        ]

        original_lower = original_command.lower()
        return any(indicator in original_lower for indicator in command_indicators)

    def auto_select_provider_and_model(self, command: str) -> tuple:
        """Auto-select best provider and model for the command"""
        try:
            if not self.api_manager or not hasattr(self.api_manager, 'api_configs'):
                return None, None

            # Define provider priorities and capabilities
            provider_priority = {
                APIProvider.OPENAI: {"priority": 10, "models": ["gpt-4", "gpt-3.5-turbo"], "cost": 0.002},
                APIProvider.ANTHROPIC: {"priority": 9, "models": ["claude-3-opus", "claude-3-sonnet"], "cost": 0.003},
                APIProvider.GOOGLE_GEMINI: {"priority": 8, "models": ["gemini-pro"], "cost": 0.001},
                APIProvider.COHERE: {"priority": 7, "models": ["command"], "cost": 0.0015}
            }

            # Find best available provider
            best_provider = None
            best_score = -1

            for provider, config in self.api_manager.api_configs.items():
                if not config.enabled or not config.api_key:
                    continue

                if provider in provider_priority:
                    priority_data = provider_priority[provider]

                    # Calculate score based on priority, cost, and recent performance
                    score = priority_data["priority"]

                    # Boost score for recently used providers
                    time_since_use = time.time() - config.last_used
                    if time_since_use < 300:  # Used within 5 minutes
                        score += 3

                    # Penalize for high failure rate
                    if config.total_requests > 0:
                        failure_rate = config.failed_requests / config.total_requests
                        score -= failure_rate * 5

                    if score > best_score:
                        best_score = score
                        best_provider = provider

            if best_provider:
                config = self.api_manager.api_configs[best_provider]
                priority_data = provider_priority[best_provider]

                # Select best model (prioritize more capable models)
                selected_model = priority_data["models"][0]  # Best model first

                # Ensure model is available in config
                if selected_model not in config.models:
                    selected_model = config.models[0] if config.models else selected_model

                return best_provider, selected_model

            return None, None

        except Exception as e:
            self.console.print(f"[bold yellow]⚠️  Error auto-selecting provider: {e}[/bold yellow]")
            return None, None

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.api_manager:
                asyncio.run(self.api_manager.shutdown())

            self.running = False

        except Exception as e:
            self.console.print(f"[bold yellow]⚠️  Error during cleanup: {e}[/bold yellow]")


def main():
    """Main entry point"""
    try:
        interface = TerminalInterface()
        interface.run()
    except KeyboardInterrupt:
        print("\nJARVIS Terminal Interface terminated")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()