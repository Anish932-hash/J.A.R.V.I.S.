#!/usr/bin/env python3
"""
J.A.R.V.I.S. Main Launcher
Advanced AI Personal Assistant for Windows
"""

import sys
import os
import time
import argparse
import logging
import warnings
from typing import Optional

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add current directory and parent directory to path for proper imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# Import JARVIS system
from jarvis.core.jarvis import JARVIS


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create logs directory
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(logs_dir, 'jarvis.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )


def print_banner():
    """Print J.A.R.V.I.S. startup banner"""
    try:
        banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    J.A.R.V.I.S. 2.0                          ║
    ║              Advanced AI Personal Assistant                  ║
    ║                                                              ║
    ║  "Sometimes you've got to run before you can walk."          ║
    ║  - Tony Stark                                                ║
    ║                                                              ║
    ║  Features:                                                   ║
    ║  • Voice Recognition & Text-to-Speech                        ║
    ║  • Real-time System Monitoring                               ║
    ║  • Application Control & Automation                          ║
    ║  • Advanced File Management                                  ║
    ║  • Network Monitoring & Control                              ║
    ║  • Security & Access Control                                 ║
    ║  • Plugin System for Extensibility                           ║
    ║  • Futuristic GUI Interface                                  ║
    ║                                                              ║
    ║  Built with Advanced Technologies for Windows                ║
    ╚══════════════════════════════════════════════════════════════╝
    """
        print(banner)
    except UnicodeEncodeError:
        # Fallback banner for systems with encoding issues
        simple_banner = """
    ==================================================
                    J.A.R.V.I.S. 2.0
              Advanced AI Personal Assistant

    "Sometimes you've got to run before you can walk."
    - Tony Stark

    Features:
    • Voice Recognition & Text-to-Speech
    • Real-time System Monitoring
    • Application Control & Automation
    • Advanced File Management
    • Network Monitoring & Control
    • Security & Access Control
    • Plugin System for Extensibility
    • Futuristic GUI Interface

    Built with Advanced Technologies for Windows
    ==================================================
    """
        print(simple_banner)


def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'psutil',
        'pywin32',
        'pyttsx3',
        'speech_recognition',
        'pyaudio'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == 'pywin32':
                # Special handling for pywin32
                import win32api
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("WARNING: Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")

        print("\nInstall missing packages:")
        print(f"   pip install -r requirements.txt")

        return False

    return True


def run_diagnostics():
    """Run system diagnostics"""
    print("Running system diagnostics...")

    try:
        import psutil

        # System info
        print("   [OK] System Info:")
        print(f"     - Platform: {psutil.sys.platform}")
        print(f"     - CPU Cores: {psutil.cpu_count()}")
        print(f"     - Memory: {psutil.virtual_memory().total // (1024**3)} GB")

        # Check for microphone
        try:
            import speech_recognition as sr
            mic = sr.Microphone()
            print("   [OK] Microphone: Available")
        except:
            print("   [WARN] Microphone: Not detected")

        # Check for speakers
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            print(f"   [OK] Text-to-Speech: Available ({len(voices)} voices)")
        except:
            print("   [WARN] Text-to-Speech: Not available")

        # Check processor info
        try:
            import platform
            processor = platform.processor()
            if processor:
                print(f"   [OK] Processor: {processor}")
            else:
                print("   [WARN] Processor: Information not available")
        except:
            print("   [WARN] Processor: Could not detect")

        print("   [OK] Diagnostics completed")

    except Exception as e:
        print(f"   [ERROR] Error during diagnostics: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. - Advanced AI Personal Assistant")
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--no-gui', action='store_true', help='Run without GUI')
    parser.add_argument('--voice-only', action='store_true', help='Voice interface only')
    parser.add_argument('--diagnostics', action='store_true', help='Run diagnostics only')
    parser.add_argument('--test-voice', action='store_true', help='Test voice interface')

    args = parser.parse_args()

    try:
        # Print banner
        print_banner()

        # Setup logging
        setup_logging(args.verbose)

        # Check requirements
        if not check_requirements():
            print("\n[ERROR] Missing requirements. Please install required packages.")
            return 1

        # Run diagnostics if requested
        if args.diagnostics:
            run_diagnostics()
            return 0

        # Test voice interface if requested
        if args.test_voice:
            from test_voice import test_voice_interface
            success = test_voice_interface()
            return 0 if success else 1

        print("Starting J.A.R.V.I.S...")

        # Create JARVIS instance
        jarvis = JARVIS(args.config)

        # Setup signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            print("\nShutting down J.A.R.V.I.S...")
            jarvis.shutdown()
            sys.exit(0)

        import signal
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start JARVIS
        try:
            import asyncio
            asyncio.run(jarvis.start())

            # Keep main thread alive
            while jarvis.is_running:
                time.sleep(1)

        except KeyboardInterrupt:
            print("\nShutdown requested by user")
        except Exception as e:
            print(f"\n[ERROR] Error running J.A.R.V.I.S.: {e}")
            return 1
        finally:
            jarvis.shutdown()

        print("J.A.R.V.I.S. shutdown complete")
        return 0

    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)