"""
J.A.R.V.I.S. Advanced GUI System
Ultra-modern PyQt6 interface with holographic effects and real-time visualizations
"""

import sys
import os
import time
import threading
import json
from typing import Optional, Dict, List, Any, Callable
import logging

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QTextEdit, QLineEdit, QProgressBar, QFrame, QSplitter,
    QTabWidget, QScrollArea, QGroupBox, QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QSpinBox,
    QCheckBox, QRadioButton, QButtonGroup, QMenuBar, QMenu, QStatusBar,
    QSystemTrayIcon, QDialog, QInputDialog, QMessageBox, QColorDialog,
    QFontDialog, QProgressDialog, QTreeWidget, QTreeWidgetItem
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve,
    QRect, QPoint, QSize, QUrl, QDateTime, QElapsedTimer
)
from PyQt6.QtGui import (
    QFont, QPalette, QColor, QBrush, QLinearGradient, QRadialGradient,
    QPainter, QPen, QIcon, QPixmap, QFontDatabase, QAction, QKeySequence,
    QPainterPath, QTransform, QMovie
)
try:
    from PyQt6.QtCharts import (
        QChart, QChartView, QLineSeries, QAreaSeries, QBarSeries, QBarSet,
        QPieSeries, QPieSlice, QValueAxis, QDateTimeAxis, QCategoryAxis
    )
    QT_CHARTS_AVAILABLE = True
except ImportError:
    QT_CHARTS_AVAILABLE = False
    # Create dummy classes for fallback
    class QChart: pass
    class QChartView: pass
    class QLineSeries: pass
    class QValueAxis: pass

# Matplotlib for advanced charts (fallback)
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class HolographicWindow(QMainWindow):
    """
    Ultra-advanced holographic GUI window for J.A.R.V.I.S.
    Features real-time visualizations, voice control, and futuristic design
    """

    # Signals for thread communication
    status_updated = pyqtSignal(dict)
    command_executed = pyqtSignal(str, dict)
    voice_command_received = pyqtSignal(str)

    def __init__(self, jarvis_instance):
        """
        Initialize the holographic main window

        Args:
            jarvis_instance: Reference to main JARVIS instance
        """
        super().__init__()
        self.jarvis = jarvis_instance
        self.logger = logging.getLogger('JARVIS.GUI')

        # Window properties
        self.setWindowTitle("J.A.R.V.I.S. 2.0 - Advanced AI Assistant")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 800)

        # Theme and styling
        self.current_theme = "futuristic_dark"
        self.animation_enabled = True
        self.holographic_effects = True

        # Data storage
        self.command_history = []
        self.system_metrics_history = []
        self.max_history_size = 1000
        self.max_command_history = 100

        # Voice interface
        self.voice_active = False
        self.voice_listening = False

        # Animation timers
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._update_animations)
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_system_status)

        # Performance monitoring
        self.performance_timer = QElapsedTimer()
        self.performance_timer.start()

        # Initialize UI
        self._setup_window()
        self._create_menus()
        self._create_status_bar()
        self._create_central_widget()
        self._setup_animations()
        self._connect_signals()

        # Load custom fonts
        self._load_fonts()

        # Apply theme
        self._apply_theme()

        self.logger.info("Holographic GUI initialized")

    def _setup_window(self):
        """Setup window properties and appearance"""
        # Set window flags for modern appearance
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinMaxButtonsHint |
            Qt.WindowType.WindowCloseButtonHint
        )

        # Set window icon
        icon_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'jarvis.ico')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # Enable drop shadows and transparency
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, False)

    def _load_fonts(self):
        """Load custom fonts for futuristic appearance"""
        try:
            # Try to load futuristic fonts
            font_dir = os.path.join(os.path.dirname(__file__), '..', 'assets', 'fonts')
            if os.path.exists(font_dir):
                for font_file in os.listdir(font_dir):
                    if font_file.endswith(('.ttf', '.otf')):
                        QFontDatabase.addApplicationFont(os.path.join(font_dir, font_file))
        except Exception as e:
            self.logger.warning(f"Could not load custom fonts: {e}")

    def _create_menus(self):
        """Create application menus"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('&File')

        exit_action = QAction('&Exit', self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu('&View')

        theme_menu = view_menu.addMenu('&Theme')
        themes = ['futuristic_dark', 'cyberpunk', 'matrix', 'neon_blue', 'plasma']
        for theme in themes:
            theme_action = QAction(theme.replace('_', ' ').title(), self)
            theme_action.triggered.connect(lambda checked, t=theme: self._change_theme(t))
            theme_menu.addAction(theme_action)

        # Tools menu
        tools_menu = menubar.addMenu('&Tools')

        diagnostics_action = QAction('&Run Diagnostics', self)
        diagnostics_action.triggered.connect(self._run_diagnostics)
        tools_menu.addAction(diagnostics_action)

        # Help menu
        help_menu = menubar.addMenu('&Help')

        about_action = QAction('&About J.A.R.V.I.S.', self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _create_status_bar(self):
        """Create status bar with real-time indicators"""
        self.status_bar = self.statusBar()

        # Status indicators
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)

        self.status_bar.addPermanentWidget(QLabel("CPU: --%"))
        self.cpu_status_label = self.status_bar.children()[-1]

        self.status_bar.addPermanentWidget(QLabel("MEM: --%"))
        self.memory_status_label = self.status_bar.children()[-1]

        self.status_bar.addPermanentWidget(QLabel("NET: --"))
        self.network_status_label = self.status_bar.children()[-1]

    def _create_central_widget(self):
        """Create the main central widget with tabbed interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        layout.addWidget(self.tab_widget)

        # Create tabs
        self._create_dashboard_tab()
        self._create_command_tab()
        self._create_monitoring_tab()
        self._create_system_control_tab()
        self._create_plugins_tab()
        self._create_security_tab()
        self._create_advanced_tab()

    def _create_dashboard_tab(self):
        """Create main dashboard tab"""
        dashboard_widget = QWidget()
        layout = QVBoxLayout(dashboard_widget)

        # Top status bar
        self._create_dashboard_header(layout)

        # Main dashboard grid
        dashboard_grid = QGridLayout()

        # System metrics cards
        self._create_system_metrics_cards(dashboard_grid)

        # Charts area
        self._create_charts_area(dashboard_grid)

        layout.addLayout(dashboard_grid)

        self.tab_widget.addTab(dashboard_widget, "Dashboard")

    def _create_dashboard_header(self, parent_layout):
        """Create dashboard header with status indicators"""
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.Shape.Box)
        header_layout = QHBoxLayout(header_frame)

        # JARVIS logo/title with glow effect
        title_label = QLabel("J.A.R.V.I.S.")
        title_font = QFont("Arial", 24, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet("""
            color: #00ffff;
            text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 30px #00ffff;
        """)
        header_layout.addWidget(title_label)

        # Version info
        version_label = QLabel("v2.0")
        version_label.setStyleSheet("color: #888888; font-size: 12px;")
        header_layout.addWidget(version_label)

        header_layout.addStretch()

        # Real-time clock
        self.clock_label = QLabel()
        self.clock_label.setStyleSheet("color: #ffffff; font-size: 14px; font-weight: bold;")
        self._update_clock()
        header_layout.addWidget(self.clock_label)

        # Status indicators with animation
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("color: #00ff00; font-size: 20px;")
        header_layout.addWidget(self.status_indicator)

        status_text = QLabel("SYSTEM ACTIVE")
        status_text.setStyleSheet("color: #ffffff; font-weight: bold;")
        header_layout.addWidget(status_text)

        # Voice status indicator
        self.voice_status_label = QLabel("🎤")
        self.voice_status_label.setStyleSheet("color: #666666; font-size: 16px;")
        self.voice_status_label.setToolTip("Voice interface status")
        header_layout.addWidget(self.voice_status_label)

        parent_layout.addWidget(header_frame)

    def _create_system_metrics_cards(self, grid_layout):
        """Create system metrics cards"""
        # CPU Card
        cpu_card = self._create_metric_card("CPU Usage", "cpu")
        grid_layout.addWidget(cpu_card, 0, 0)

        # Memory Card
        memory_card = self._create_metric_card("Memory Usage", "memory")
        grid_layout.addWidget(memory_card, 0, 1)

        # Disk Card
        disk_card = self._create_metric_card("Disk Usage", "disk")
        grid_layout.addWidget(disk_card, 0, 2)

        # Network Card
        network_card = self._create_metric_card("Network Status", "network")
        grid_layout.addWidget(network_card, 1, 0)

    def _create_metric_card(self, title, metric_type):
        """Create a metric card widget"""
        card = QGroupBox(title)
        card.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #00ffff;
                border-radius: 5px;
                margin-top: 1ex;
                color: #00ffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

        layout = QVBoxLayout(card)

        # Progress bar
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setTextVisible(True)
        layout.addWidget(progress)

        # Value label
        value_label = QLabel("--%")
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setStyleSheet("font-size: 18px; color: #ffffff;")
        layout.addWidget(value_label)

        # Store references
        if metric_type == "cpu":
            self.cpu_progress = progress
            self.cpu_value_label = value_label
        elif metric_type == "memory":
            self.memory_progress = progress
            self.memory_value_label = value_label
        elif metric_type == "disk":
            self.disk_progress = progress
            self.disk_value_label = value_label
        elif metric_type == "network":
            self.network_status_label_card = value_label

        return card

    def _create_charts_area(self, grid_layout):
        """Create charts area with real-time visualizations"""
        charts_group = QGroupBox("Real-time Monitoring")
        charts_layout = QVBoxLayout(charts_group)

        # Chart tabs
        chart_tabs = QTabWidget()

        # CPU Chart
        cpu_chart = self._create_line_chart("CPU Usage Over Time")
        chart_tabs.addTab(cpu_chart, "CPU")

        # Memory Chart
        memory_chart = self._create_line_chart("Memory Usage Over Time")
        chart_tabs.addTab(memory_chart, "Memory")

        # Network Chart
        network_chart = self._create_line_chart("Network Activity")
        chart_tabs.addTab(network_chart, "Network")

        charts_layout.addWidget(chart_tabs)
        grid_layout.addWidget(charts_group, 1, 1, 1, 2)

    def _create_line_chart(self, title):
        """Create a line chart for monitoring"""
        if not QT_CHARTS_AVAILABLE:
            # Fallback to a simple widget with text
            from PyQt6.QtWidgets import QLabel
            fallback_widget = QLabel(f"{title}\n(Charts not available - install PyQt6-Charts)")
            fallback_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback_widget.setStyleSheet("color: #ffff00; font-size: 14px; padding: 20px;")
            return fallback_widget

        chart = QChart()
        chart.setTitle(title)
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)

        series = QLineSeries()
        series.setName("Usage %")

        chart.addSeries(series)

        axis_x = QValueAxis()
        axis_x.setRange(0, 60)  # 60 seconds
        axis_x.setTitleText("Time (seconds)")

        axis_y = QValueAxis()
        axis_y.setRange(0, 100)
        axis_y.setTitleText("Usage (%)")

        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_x)
        series.attachAxis(axis_y)

        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)

        return chart_view

    def _create_command_tab(self):
        """Create command interface tab"""
        command_widget = QWidget()
        layout = QVBoxLayout(command_widget)

        # Command input area
        input_group = QGroupBox("Command Input")
        input_layout = QHBoxLayout(input_group)

        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("Enter command or speak...")
        self.command_input.returnPressed.connect(self._execute_command)
        input_layout.addWidget(self.command_input)

        execute_btn = QPushButton("Execute")
        execute_btn.clicked.connect(self._execute_command)
        input_layout.addWidget(execute_btn)

        voice_btn = QPushButton("🎤 Voice")
        voice_btn.setCheckable(True)
        voice_btn.clicked.connect(self._toggle_voice_mode)
        input_layout.addWidget(voice_btn)

        layout.addWidget(input_group)

        # Output area
        output_group = QGroupBox("Command Output")
        output_layout = QVBoxLayout(output_group)

        self.command_output = QTextEdit()
        self.command_output.setReadOnly(True)
        self.command_output.setStyleSheet("""
            QTextEdit {
                background-color: #000000;
                color: #00ff00;
                font-family: 'Courier New';
                font-size: 10px;
            }
        """)
        output_layout.addWidget(self.command_output)

        layout.addWidget(output_group)

        # Command history
        history_group = QGroupBox("Command History")
        history_layout = QVBoxLayout(history_group)

        self.command_history_list = QListWidget()
        self.command_history_list.itemDoubleClicked.connect(self._reuse_command)
        history_layout.addWidget(self.command_history_list)

        layout.addWidget(history_group)

        self.tab_widget.addTab(command_widget, "Commands")

    def _create_monitoring_tab(self):
        """Create advanced monitoring tab"""
        monitoring_widget = QWidget()
        layout = QVBoxLayout(monitoring_widget)

        # Process monitor
        process_group = QGroupBox("Process Monitor")
        process_layout = QVBoxLayout(process_group)

        self.process_table = QTableWidget()
        self.process_table.setColumnCount(4)
        self.process_table.setHorizontalHeaderLabels(["PID", "Name", "CPU %", "Memory %"])
        self.process_table.horizontalHeader().setStretchLastSection(True)
        process_layout.addWidget(self.process_table)

        layout.addWidget(process_group)

        # Network monitor
        network_group = QGroupBox("Network Monitor")
        network_layout = QVBoxLayout(network_group)

        self.network_table = QTableWidget()
        self.network_table.setColumnCount(3)
        self.network_table.setHorizontalHeaderLabels(["Interface", "IP Address", "Status"])
        self.network_table.horizontalHeader().setStretchLastSection(True)
        network_layout.addWidget(self.network_table)

        layout.addWidget(network_group)

        self.tab_widget.addTab(monitoring_widget, "Monitoring")

    def _create_system_control_tab(self):
        """Create system control tab"""
        control_widget = QWidget()
        layout = QVBoxLayout(control_widget)

        # Service control
        service_group = QGroupBox("Windows Services")
        service_layout = QVBoxLayout(service_group)

        self.service_table = QTableWidget()
        self.service_table.setColumnCount(3)
        self.service_table.setHorizontalHeaderLabels(["Service", "Status", "Startup Type"])
        self.service_table.horizontalHeader().setStretchLastSection(True)
        service_layout.addWidget(self.service_table)

        # Control buttons
        service_buttons = QHBoxLayout()
        start_btn = QPushButton("Start Service")
        stop_btn = QPushButton("Stop Service")
        restart_btn = QPushButton("Restart Service")

        service_buttons.addWidget(start_btn)
        service_buttons.addWidget(stop_btn)
        service_buttons.addWidget(restart_btn)
        service_layout.addLayout(service_buttons)

        layout.addWidget(service_group)

        # Startup programs
        startup_group = QGroupBox("Startup Programs")
        startup_layout = QVBoxLayout(startup_group)

        self.startup_table = QTableWidget()
        self.startup_table.setColumnCount(2)
        self.startup_table.setHorizontalHeaderLabels(["Program", "Location"])
        self.startup_table.horizontalHeader().setStretchLastSection(True)
        startup_layout.addWidget(self.startup_table)

        layout.addWidget(startup_group)

        self.tab_widget.addTab(control_widget, "System Control")

    def _create_plugins_tab(self):
        """Create plugin management tab"""
        plugins_widget = QWidget()
        layout = QVBoxLayout(plugins_widget)

        # Plugin list
        plugin_group = QGroupBox("Installed Plugins")
        plugin_layout = QVBoxLayout(plugin_group)

        self.plugin_table = QTableWidget()
        self.plugin_table.setColumnCount(4)
        self.plugin_table.setHorizontalHeaderLabels(["Name", "Version", "Status", "Description"])
        self.plugin_table.horizontalHeader().setStretchLastSection(True)
        plugin_layout.addWidget(self.plugin_table)

        # Plugin controls
        plugin_controls = QHBoxLayout()
        load_btn = QPushButton("Load Plugin")
        unload_btn = QPushButton("Unload Plugin")
        install_btn = QPushButton("Install from File")

        plugin_controls.addWidget(load_btn)
        plugin_controls.addWidget(unload_btn)
        plugin_controls.addWidget(install_btn)
        plugin_layout.addLayout(plugin_controls)

        layout.addWidget(plugin_group)

        # Plugin marketplace (if available)
        marketplace_group = QGroupBox("Plugin Marketplace")
        marketplace_layout = QVBoxLayout(marketplace_group)

        self.marketplace_table = QTableWidget()
        self.marketplace_table.setColumnCount(3)
        self.marketplace_table.setHorizontalHeaderLabels(["Plugin", "Category", "Rating"])
        self.marketplace_table.horizontalHeader().setStretchLastSection(True)
        marketplace_layout.addWidget(self.marketplace_table)

        download_btn = QPushButton("Download & Install")
        marketplace_layout.addWidget(download_btn)

        layout.addWidget(marketplace_group)

        self.tab_widget.addTab(plugins_widget, "Plugins")

    def _create_security_tab(self):
        """Create security monitoring tab"""
        security_widget = QWidget()
        layout = QVBoxLayout(security_widget)

        # Threat monitor
        threat_group = QGroupBox("Security Threats")
        threat_layout = QVBoxLayout(threat_group)

        self.threat_table = QTableWidget()
        self.threat_table.setColumnCount(4)
        self.threat_table.setHorizontalHeaderLabels(["Time", "Type", "Severity", "Description"])
        self.threat_table.horizontalHeader().setStretchLastSection(True)
        threat_layout.addWidget(self.threat_table)

        layout.addWidget(threat_group)

        # Access log
        access_group = QGroupBox("Access Log")
        access_layout = QVBoxLayout(access_group)

        self.access_log = QTextEdit()
        self.access_log.setReadOnly(True)
        access_layout.addWidget(self.access_log)

        layout.addWidget(access_group)

        # Security controls
        controls_group = QGroupBox("Security Controls")
        controls_layout = QGridLayout(controls_group)

        self.firewall_checkbox = QCheckBox("Windows Firewall")
        self.antivirus_checkbox = QCheckBox("Windows Defender")
        self.encryption_checkbox = QCheckBox("Data Encryption")

        controls_layout.addWidget(self.firewall_checkbox, 0, 0)
        controls_layout.addWidget(self.antivirus_checkbox, 0, 1)
        controls_layout.addWidget(self.encryption_checkbox, 1, 0)

        scan_btn = QPushButton("Run Security Scan")
        controls_layout.addWidget(scan_btn, 1, 1)

        layout.addWidget(controls_group)

        self.tab_widget.addTab(security_widget, "Security")

    def _create_advanced_tab(self):
        """Create advanced features tab"""
        advanced_widget = QWidget()
        layout = QVBoxLayout(advanced_widget)

        # Self-development engine
        dev_group = QGroupBox("Self-Development Engine")
        dev_layout = QVBoxLayout(dev_group)

        self.dev_status_label = QLabel("Status: Inactive")
        dev_layout.addWidget(self.dev_status_label)

        dev_controls = QHBoxLayout()
        start_dev_btn = QPushButton("Start Development")
        stop_dev_btn = QPushButton("Stop Development")
        dev_stats_btn = QPushButton("View Statistics")

        dev_controls.addWidget(start_dev_btn)
        dev_controls.addWidget(stop_dev_btn)
        dev_controls.addWidget(dev_stats_btn)
        dev_layout.addLayout(dev_controls)

        layout.addWidget(dev_group)

        # Application healer
        healer_group = QGroupBox("Application Healer")
        healer_layout = QVBoxLayout(healer_group)

        self.healer_status_label = QLabel("Status: Monitoring")
        healer_layout.addWidget(self.healer_status_label)

        healer_btn = QPushButton("Run Health Check")
        healer_layout.addWidget(healer_btn)

        layout.addWidget(healer_group)

        # IoT Integration
        iot_group = QGroupBox("IoT Integration")
        iot_layout = QVBoxLayout(iot_group)

        self.iot_status_label = QLabel("Connected Devices: 0")
        iot_layout.addWidget(self.iot_status_label)

        iot_btn = QPushButton("Scan for Devices")
        iot_layout.addWidget(iot_btn)

        layout.addWidget(iot_group)

        self.tab_widget.addTab(advanced_widget, "Advanced")

    def _setup_animations(self):
        """Setup GUI animations"""
        if self.animation_enabled:
            self.animation_timer.start(50)  # 20 FPS

    def _connect_signals(self):
        """Connect signals and slots"""
        self.status_updated.connect(self._on_status_updated)
        self.command_executed.connect(self._on_command_executed)
        self.voice_command_received.connect(self._on_voice_command_received)

    def _apply_theme(self):
        """Apply the current theme"""
        if self.current_theme == "futuristic_dark":
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #0a0a0a;
                    color: #ffffff;
                }

                QTabWidget::pane {
                    border: 1px solid #00ffff;
                    background-color: #1a1a1a;
                }

                QTabBar::tab {
                    background-color: #2a2a2a;
                    color: #ffffff;
                    padding: 8px 16px;
                    border: 1px solid #00ffff;
                    border-bottom: none;
                }

                QTabBar::tab:selected {
                    background-color: #00ffff;
                    color: #000000;
                }

                QGroupBox {
                    font-weight: bold;
                    border: 2px solid #00ffff;
                    border-radius: 5px;
                    margin-top: 1ex;
                    color: #00ffff;
                }

                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }

                QPushButton {
                    background-color: #2a2a2a;
                    color: #ffffff;
                    border: 1px solid #00ffff;
                    border-radius: 3px;
                    padding: 5px 10px;
                }

                QPushButton:hover {
                    background-color: #00ffff;
                    color: #000000;
                }

                QPushButton:pressed {
                    background-color: #0080ff;
                }

                QLineEdit, QTextEdit {
                    background-color: #000000;
                    color: #00ff00;
                    border: 1px solid #00ffff;
                    border-radius: 3px;
                    padding: 2px;
                }

                QProgressBar {
                    border: 1px solid #00ffff;
                    border-radius: 3px;
                    text-align: center;
                }

                QProgressBar::chunk {
                    background-color: #00ffff;
                }

                QListWidget, QTableWidget {
                    background-color: #000000;
                    color: #00ff00;
                    border: 1px solid #00ffff;
                    gridline-color: #00ffff;
                }

                QHeaderView::section {
                    background-color: #2a2a2a;
                    color: #ffffff;
                    border: 1px solid #00ffff;
                    padding: 4px;
                }

                QStatusBar {
                    background-color: #2a2a2a;
                    color: #ffffff;
                    border-top: 1px solid #00ffff;
                }
            """)

    def _change_theme(self, theme_name):
        """Change the application theme"""
        self.current_theme = theme_name
        self._apply_theme()

    def _update_animations(self):
        """Update GUI animations"""
        if not self.animation_enabled:
            return

        # Pulse effect for status indicator
        import math
        pulse = (math.sin(time.time() * 2) + 1) / 2  # 0 to 1

        if hasattr(self, 'status_indicator'):
            # Color cycling
            hue = (time.time() * 50) % 360
            color = QColor.fromHsv(int(hue), 255, 255)
            self.status_indicator.setStyleSheet(f"color: {color.name()}; font-size: 20px;")

    def _update_clock(self):
        """Update the real-time clock display"""
        try:
            from datetime import datetime
            current_time = datetime.now().strftime("%H:%M:%S")
            if hasattr(self, 'clock_label'):
                self.clock_label.setText(f"🕐 {current_time}")
        except Exception as e:
            self.logger.error(f"Error updating clock: {e}")

    def _update_system_status(self):
        """Update system status displays"""
        try:
            if not self.jarvis:
                return

            # Update clock
            self._update_clock()

            # Get system status
            status = self.jarvis.get_status()

            # Update CPU
            if self.jarvis.system_monitor:
                cpu_info = self.jarvis.system_monitor.current_readings.get('cpu', {})
                cpu_percent = cpu_info.get('percent', 0)

                if hasattr(self, 'cpu_progress'):
                    self.cpu_progress.setValue(int(cpu_percent))
                if hasattr(self, 'cpu_value_label'):
                    self.cpu_value_label.setText(f"{cpu_percent:.1f}%")
                if hasattr(self, 'cpu_status_label'):
                    self.cpu_status_label.setText(f"CPU: {cpu_percent:.1f}%")

            # Update Memory
            if self.jarvis.system_monitor:
                memory_info = self.jarvis.system_monitor.current_readings.get('memory', {})
                memory_percent = memory_info.get('percent', 0)

                if hasattr(self, 'memory_progress'):
                    self.memory_progress.setValue(int(memory_percent))
                if hasattr(self, 'memory_value_label'):
                    self.memory_value_label.setText(f"{memory_percent:.1f}%")
                if hasattr(self, 'memory_status_label'):
                    self.memory_status_label.setText(f"MEM: {memory_percent:.1f}%")

            # Update Disk
            if self.jarvis.system_monitor:
                disk_info = self.jarvis.system_monitor.current_readings.get('disk', {})
                disk_percent = disk_info.get('main_percent', 0)

                if hasattr(self, 'disk_progress'):
                    self.disk_progress.setValue(int(disk_percent))
                if hasattr(self, 'disk_value_label'):
                    self.disk_value_label.setText(f"{disk_percent:.1f}%")

            # Update Network
            network_status = "Connected"
            if hasattr(self, 'network_status_label'):
                self.network_status_label.setText(f"NET: {network_status}")
            if hasattr(self, 'network_status_label_card'):
                self.network_status_label_card.setText(network_status)

            # Update voice status
            if hasattr(self, 'voice_status_label'):
                if self.jarvis.voice_interface and self.jarvis.voice_interface.listening:
                    self.voice_status_label.setStyleSheet("color: #00ff00; font-size: 16px;")
                    self.voice_status_label.setText("🎤")
                    self.voice_status_label.setToolTip("Voice interface: ACTIVE")
                else:
                    self.voice_status_label.setStyleSheet("color: #666666; font-size: 16px;")
                    self.voice_status_label.setText("🎤")
                    self.voice_status_label.setToolTip("Voice interface: INACTIVE")

            # Update status indicator with pulsing effect
            if hasattr(self, 'status_indicator'):
                import time
                pulse_intensity = int(128 + 127 * abs(time.time() % 2 - 1))  # Breathing effect
                self.status_indicator.setStyleSheet(f"color: rgb(0, {pulse_intensity}, 0); font-size: 20px;")

            # Update active modules
            if 'active_modules' in status:
                modules_text = "\n".join(f"• {module}" for module in status['active_modules'])
                # Update modules display if exists

        except Exception as e:
            self.logger.error(f"Error updating system status: {e}")

    def _execute_command(self):
        """Execute user command"""
        try:
            command = self.command_input.text().strip()
            if not command:
                return

            # Add to history
            self._add_to_command_history(command)

            # Clear input
            self.command_input.clear()

            # Execute command
            if self.jarvis:
                result = self.jarvis.execute_command(command, {"source": "gui"})

                # Display result
                self._display_command_result(command, result)

        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            self._display_command_result("ERROR", {"error": str(e)})

    def _toggle_voice_mode(self):
        """Toggle voice command mode"""
        try:
            if self.jarvis and self.jarvis.voice_interface:
                if self.voice_listening:
                    self.jarvis.voice_interface.stop_continuous_listening()
                    self.voice_listening = False
                    self.sender().setText("🎤 Voice")
                    self.sender().setChecked(False)
                else:
                    self.jarvis.voice_interface.start_continuous_listening()
                    self.voice_listening = True
                    self.sender().setText("🎤🔴 Listening")
                    self.sender().setChecked(True)

        except Exception as e:
            self.logger.error(f"Error toggling voice mode: {e}")

    def _add_to_command_history(self, command):
        """Add command to history"""
        try:
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            history_item = f"[{timestamp}] {command}"

            self.command_history.append(history_item)
            if len(self.command_history) > self.max_command_history:
                self.command_history.pop(0)

            # Update history list
            if hasattr(self, 'command_history_list'):
                self.command_history_list.clear()
                for item in reversed(self.command_history):
                    self.command_history_list.addItem(item)

        except Exception as e:
            self.logger.error(f"Error adding to command history: {e}")

    def _display_command_result(self, command, result):
        """Display command execution result"""
        try:
            timestamp = time.strftime("%H:%M:%S", time.localtime())

            # Format result
            if "error" in result:
                result_text = f"❌ ERROR: {result['error']}"
                color = "#ff0000"
            elif "success" in result and result["success"]:
                if "message" in result:
                    result_text = f"✅ {result['message']}"
                else:
                    result_text = "✅ Command executed successfully"
                color = "#00ff00"
            else:
                result_text = f"ℹ️  {result}"
                color = "#ffff00"

            # Add to output
            current_text = self.command_output.toPlainText()
            new_text = f"[{timestamp}] > {command}\n{result_text}\n\n"
            self.command_output.setPlainText(current_text + new_text)

            # Auto-scroll to bottom
            scrollbar = self.command_output.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        except Exception as e:
            self.logger.error(f"Error displaying command result: {e}")

    def _reuse_command(self, item):
        """Reuse a command from history"""
        try:
            command_text = item.text().split("] ", 1)[1] if "] " in item.text() else item.text()
            self.command_input.setText(command_text)
        except Exception as e:
            self.logger.error(f"Error reusing command: {e}")

    def _run_diagnostics(self):
        """Run system diagnostics"""
        try:
            if self.jarvis and self.jarvis.system_monitor:
                # Get system info
                system_info = self.jarvis.system_monitor.get_system_info()

                # Display results
                result_text = "System Diagnostics Results:\n"
                for key, value in system_info.items():
                    result_text += f"• {key}: {value}\n"

                self._display_command_result("SYSTEM DIAGNOSTICS", {"result": result_text})

        except Exception as e:
            self.logger.error(f"Error running diagnostics: {e}")

    def _show_about(self):
        """Show about dialog"""
        about_text = """
        J.A.R.V.I.S. 2.0 - Advanced AI Personal Assistant

        Ultra-advanced AI assistant for complete Windows PC control and automation.

        Features:
        • Voice Recognition & Text-to-Speech
        • Real-time System Monitoring
        • Application Control & Automation
        • Advanced File Management
        • Network Monitoring & Control
        • Security & Access Control
        • Plugin System for Extensibility
        • Futuristic GUI Interface
        • Self-Development Engine
        • Application Healer
        • IoT Integration

        Built with cutting-edge technologies for the future.
        """
        QMessageBox.about(self, "About J.A.R.V.I.S.", about_text)

    def _on_status_updated(self, status):
        """Handle status update signal"""
        # Update UI with new status
        pass

    def _on_command_executed(self, command, result):
        """Handle command execution signal"""
        self._display_command_result(command, result)

    def _on_voice_command_received(self, command):
        """Handle voice command received signal"""
        self.command_input.setText(command)
        self._execute_command()

    def show(self):
        """Show the main window"""
        self.showMaximized()
        self.status_timer.start(1000)  # Update every second
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(1000)  # Update clock every second

    def closeEvent(self, event):
        """Handle window close event"""
        self.animation_timer.stop()
        self.status_timer.stop()
        if hasattr(self, 'clock_timer'):
            self.clock_timer.stop()
        event.accept()

class JARVISGUI:
    """Main GUI controller class for the advanced holographic interface"""

    def __init__(self, jarvis_instance):
        """
        Initialize GUI controller

        Args:
            jarvis_instance: Reference to main JARVIS instance
        """
        self.jarvis = jarvis_instance
        self.main_window = None

    def show(self):
        """Show the advanced holographic GUI"""
        try:
            # Create QApplication if it doesn't exist
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)

            self.main_window = HolographicWindow(self.jarvis)
            self.main_window.show()

            # Start the application event loop
            sys.exit(app.exec())

        except Exception as e:
            logging.error(f"Error showing advanced GUI: {e}")
            raise

    def hide(self):
        """Hide the GUI"""
        try:
            if self.main_window:
                self.main_window.close()

        except Exception as e:
            logging.error(f"Error hiding GUI: {e}")