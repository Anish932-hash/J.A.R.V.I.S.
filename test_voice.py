#!/usr/bin/env python3
"""
J.A.R.V.I.S. Voice Interface Test
Test script for voice recognition and text-to-speech
"""

import sys
import os
import time

# Add jarvis to path
sys.path.insert(0, os.path.dirname(__file__))

from jarvis.modules.voice_interface import VoiceInterface


class MockJARVIS:
    """Mock JARVIS instance for testing"""

    def __init__(self):
        self.logger = type('MockLogger', (), {
            'info': lambda msg: print(f"INFO: {msg}"),
            'error': lambda msg: print(f"ERROR: {msg}"),
            'debug': lambda msg: print(f"DEBUG: {msg}"),
            'warning': lambda msg: print(f"WARNING: {msg}")
        })()

    def execute_command(self, command, context=None):
        print(f"Executing command: {command}")
        return {"success": True, "message": f"Executed: {command}"}


def test_voice_interface():
    """Test voice interface functionality"""
    print("Testing J.A.R.V.I.S. Voice Interface...")
    print("=" * 50)

    # Create mock JARVIS instance
    mock_jarvis = MockJARVIS()

    # Create voice interface
    voice = VoiceInterface(mock_jarvis)

    try:
        # Initialize voice interface
        print("1. Initializing voice interface...")
        voice.initialize()
        print("   ✓ Voice interface initialized")

        # Test TTS
        print("\n2. Testing text-to-speech...")
        test_text = "Hello! This is J.A.R.V.I.S. voice interface test."
        voice.speak(test_text)
        print("   ✓ Speech test completed")

        # Test voice settings
        print("\n3. Testing voice settings...")
        voices = voice.get_available_voices()
        print(f"   Found {len(voices)} voices")

        if voices:
            print(f"   Current voice: {voices[0]['name'] if voices else 'None'}")

        # Test speech rate and volume
        voice.set_speech_rate(180)
        voice.set_speech_volume(0.7)
        print("   ✓ Voice settings configured")

        # Test STT (if available)
        print("\n4. Testing speech recognition...")
        print("   Say something in 5 seconds...")
        result = voice.listen(timeout=5)

        if result:
            print(f"   ✓ Recognized: {result}")
        else:
            print("   - No speech detected (this is normal)")

        # Test microphone calibration
        print("\n5. Testing microphone calibration...")
        if voice.calibrate_microphone(duration=2):
            print("   ✓ Microphone calibrated")
        else:
            print("   - Microphone calibration failed")

        # Get statistics
        print("\n6. Voice interface statistics:")
        stats = voice.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")

        print("\n" + "=" * 50)
        print("Voice interface test completed!")
        print("\nNote: For full functionality, ensure you have:")
        print("- Microphone connected and working")
        print("- Speakers or headphones connected")
        print("- Required packages installed (see requirements.txt)")
        print("- Internet connection for Google Speech Recognition")

        return True

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print("\nTroubleshooting:")
        print("- Make sure you have a working microphone")
        print("- Install required packages: pip install -r requirements.txt")
        print("- Check that speakers/headphones are connected")
        print("- Try running as administrator")
        return False

    finally:
        # Cleanup
        try:
            voice.shutdown()
        except:
            pass


if __name__ == "__main__":
    success = test_voice_interface()
    sys.exit(0 if success else 1)