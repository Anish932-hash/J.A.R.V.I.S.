#!/usr/bin/env python3
"""
J.A.R.V.I.S. Launcher Script
Quick launcher for different JARVIS modes
"""

import sys
import os
import argparse

# Add current directory to path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)


def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. Launcher")
    parser.add_argument('--gui', action='store_true', help='Launch with GUI')
    parser.add_argument('--terminal', action='store_true', help='Launch terminal-only mode with rich interface')
    parser.add_argument('--voice-only', action='store_true', help='Voice interface only')
    parser.add_argument('--test-voice', action='store_true', help='Test voice interface')
    parser.add_argument('--diagnostics', action='store_true', help='Run diagnostics')
    parser.add_argument('--config', help='Configuration file path')

    args = parser.parse_args()

    try:
        # Import JARVIS
        from jarvis.core.jarvis import JARVIS

        print("Launching J.A.R.V.I.S...")

        # Create JARVIS instance
        jarvis = JARVIS(args.config)

        if args.test_voice:
            # Test voice interface
            print("Testing voice interface...")
            from test_voice import test_voice_interface
            success = test_voice_interface()
            return 0 if success else 1

        elif args.diagnostics:
            # Run diagnostics
            print("Running diagnostics...")
            import asyncio
            asyncio.run(jarvis.start())

            # Get system info
            if jarvis.system_monitor:
                system_info = jarvis.system_monitor.get_system_info()
                print("System Information:")
                for key, value in system_info.items():
                    print(f"  {key}: {value}")

            jarvis.shutdown()
            return 0

        elif args.voice_only:
            # Voice-only mode
            print("Starting voice-only mode...")
            import asyncio
            asyncio.run(jarvis.start())

            try:
                # Start voice interface
                if jarvis.voice_interface:
                    jarvis.voice_interface.start_continuous_listening()

                print("Listening for voice commands... (Press Ctrl+C to exit)")

                # Keep running
                while jarvis.is_running:
                    import time
                    time.sleep(1)

            except KeyboardInterrupt:
                print("Shutting down...")
            finally:
                jarvis.shutdown()

        elif args.terminal:
            # Terminal-only mode with rich interface
            print("Starting terminal mode...")
            try:
                from terminal_interface import main as terminal_main
                terminal_main()
            except ImportError as e:
                print(f"Terminal interface not available: {e}")
                print("Make sure 'rich' library is installed: pip install rich")
                return 1

        elif args.gui:
            # GUI mode
            print("Starting GUI mode...")
            import asyncio
            asyncio.run(jarvis.start())

            try:
                # Import and show GUI
                from gui.main_window import JARVISGUI

                gui = JARVISGUI(jarvis)
                gui.show()

            except ImportError as e:
                print(f"GUI not available: {e}")
                print("Running in console mode...")
                while jarvis.is_running:
                    import time
                    time.sleep(1)
            finally:
                jarvis.shutdown()

        else:
            # Default console mode
            print("Starting console mode...")
            import asyncio
            asyncio.run(jarvis.start())

            try:
                print("J.A.R.V.I.S. is running!")
                print("Commands:")
                print("  status         - Show system status")
                print("  test voice     - Test voice interface")
                print("  system scan    - Scan system")
                print("  quit           - Shutdown")
                print()

                while jarvis.is_running:
                    try:
                        command = input("JARVIS> ").strip()
                        if command.lower() in ['quit', 'exit', 'shutdown']:
                            break

                        if command:
                            result = jarvis.execute_command(command)
                            if result:
                                print(f"Result: {result}")

                    except KeyboardInterrupt:
                        break

            except Exception as e:
                print(f"Error in console mode: {e}")
            finally:
                print("Shutting down...")
                jarvis.shutdown()

        return 0

    except Exception as e:
        print(f"Error launching J.A.R.V.I.S.: {e}")
        if args.gui:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)