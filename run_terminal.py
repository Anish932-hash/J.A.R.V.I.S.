#!/usr/bin/env python3
"""
J.A.R.V.I.S. Terminal Mode Launcher
Launch JARVIS in terminal-only mode with rich text interface
"""

import sys
import os

# Add JARVIS to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def main():
    """Launch JARVIS in terminal mode"""
    try:
        from terminal_interface import main as terminal_main
        terminal_main()
    except ImportError as e:
        print(f"Failed to import terminal interface: {e}")
        print("Make sure 'rich' library is installed: pip install rich")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nJARVIS Terminal Mode terminated")
    except Exception as e:
        print(f"Fatal error in terminal mode: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()