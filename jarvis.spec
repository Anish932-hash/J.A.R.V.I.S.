# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('config', 'config'), ('assets', 'assets'), ('docs', 'docs'), ('plugins', 'plugins')],
    hiddenimports=['jarvis.core.jarvis', 'jarvis.core.system_core', 'jarvis.core.event_manager', 'jarvis.core.command_processor', 'jarvis.modules.voice_interface', 'jarvis.modules.system_monitor', 'jarvis.modules.application_controller', 'jarvis.modules.file_manager', 'jarvis.modules.network_manager', 'jarvis.modules.security_manager', 'jarvis.modules.plugin_manager', 'jarvis.core.advanced.self_development_engine', 'jarvis.core.advanced.application_healer', 'jarvis.core.advanced.code_generator', 'jarvis.core.advanced.web_searcher', 'jarvis.core.advanced.reasoning_engine', 'jarvis.core.advanced.tester', 'jarvis.core.advanced.updater', 'jarvis.core.advanced.evolver', 'jarvis.core.advanced.validator', 'jarvis.core.advanced.info_collector', 'jarvis.core.advanced.memory_manager', 'jarvis.core.advanced.ethics_engine', 'jarvis.core.advanced.healer_components.error_detector', 'jarvis.core.advanced.healer_components.debugger', 'jarvis.core.advanced.healer_components.fix_generator', 'jarvis.core.advanced.healer_components.health_reporter', 'jarvis.core.advanced.healer_components.optimizer', 'jarvis.core.advanced.healer_components.patch_applier', 'jarvis.core.advanced.healer_components.predictor', 'jarvis.core.advanced.healer_components.recovery_manager', 'jarvis.gui.main_window', 'PyQt6', 'PyQt6.QtWidgets', 'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtCharts', 'psutil', 'pywin32', 'pyttsx3', 'speech_recognition', 'pyaudio', 'opencv_python', 'Pillow', 'pytesseract', 'mss', 'requests', 'websocket_client', 'selenium', 'cryptography', 'torch', 'transformers', 'GPUtil', 'numpy', 'pandas', 'sqlalchemy', 'colorama', 'tqdm', 'pyyaml', 'speedtest', 'rarfile', 'matplotlib', 'scipy'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'unittest', 'pdb', 'pydoc'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='JARVIS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=true,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/jarvis.ico',
    version_file=None,
)
