#!/usr/bin/env python3
"""
J.A.R.V.I.S. GUI Launcher
Advanced GUI launcher with multiple interface options
"""

import sys
import os
import argparse
import logging

# Add JARVIS to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from core.jarvis import JARVIS
    from gui.main_window import JARVISGUI
    from gui.advanced_gui import create_advanced_gui
    FULL_JARVIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Full JARVIS not available: {e}")
    print("GUI will run in demo mode")
    FULL_JARVIS_AVAILABLE = False
    JARVIS = None
    JARVISGUI = None
    create_advanced_gui = None


def setup_logging():
    """Setup logging for GUI launcher"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('jarvis_gui.log'),
            logging.StreamHandler()
        ]
    )


def launch_basic_gui(jarvis_instance):
    """Launch basic holographic GUI"""
    try:
        print("Launching J.A.R.V.I.S. Basic GUI...")
        if JARVISGUI is None:
            print("Basic GUI not available")
            return False
        gui = JARVISGUI(jarvis_instance)
        gui.show()
        return True
    except Exception as e:
        print(f"Failed to launch basic GUI: {e}")
        return False


def launch_advanced_gui(jarvis_instance):
    """Launch advanced GUI with 3D visualizations"""
    try:
        print("Launching J.A.R.V.I.S. Advanced GUI...")
        if create_advanced_gui is None:
            print("Advanced GUI not available")
            return False

        result = create_advanced_gui(jarvis_instance)

        if result:
            main_window, app = result
            main_window.show()
            print("Advanced GUI launched successfully!")
            print("Use the interface to control J.A.R.V.I.S.")
            print("Features: 3D monitoring, voice control, real-time analytics")
            return True
        else:
            print("Failed to create advanced GUI")
            return False

    except Exception as e:
        print(f"Failed to launch advanced GUI: {e}")
        return False


def main():
    """Main GUI launcher function"""
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. GUI Launcher")
    parser.add_argument('--config', '-c', default='config/jarvis.json',
                       help='Configuration file path')
    parser.add_argument('--advanced', '-a', action='store_true',
                       help='Launch advanced GUI with 3D visualizations')
    parser.add_argument('--basic', '-b', action='store_true',
                       help='Launch basic holographic GUI')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Enable debug logging')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    setup_logging()

    print("J.A.R.V.I.S. GUI Launcher")
    print("=" * 50)

    try:
        jarvis = None
        if FULL_JARVIS_AVAILABLE:
            # Initialize JARVIS
            print("Initializing J.A.R.V.I.S. core...")
            jarvis = JARVIS(args.config)

            if not jarvis:
                print("Failed to initialize JARVIS")
                return 1

            print("JARVIS core initialized")
        else:
            print("Running in demo mode - GUI only")
            jarvis = None

        # Determine which GUI to launch
        if args.advanced:
            success = launch_advanced_gui(jarvis)
        elif args.basic:
            success = launch_basic_gui(jarvis)
        else:
            # Auto-detect best available GUI
            try:
                from gui.advanced_gui import PYQT6_AVAILABLE, OPENGL_AVAILABLE
                if PYQT6_AVAILABLE:
                    print("PyQt6 detected, launching advanced GUI...")
                    success = launch_advanced_gui(jarvis)
                else:
                    print("PyQt6 not available, falling back to basic GUI...")
                    success = launch_basic_gui(jarvis)
            except ImportError:
                print("Advanced GUI not available, using basic interface...")
                success = launch_basic_gui(jarvis)

        if success:
            print("\nGUI launched successfully!")
            print("Use the interface to interact with J.A.R.V.I.S.")
            print("Try voice commands like 'JARVIS, what time is it?'")
            print("Monitor system performance in real-time")
            print("Access advanced features through the tabs")
            return 0
        else:
            print("\nFailed to launch GUI")
            return 1

    except KeyboardInterrupt:
        print("\nGUI launcher interrupted by user")
        return 0
    except Exception as e:
        print(f"\nError in GUI launcher: {e}")
        logging.exception("GUI launcher error")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)