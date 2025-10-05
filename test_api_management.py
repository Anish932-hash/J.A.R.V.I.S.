#!/usr/bin/env python3
"""
J.A.R.V.I.S. API Management Demonstration
Shows automatic API key detection and provider configuration
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from core.api_manager import APIKeyDetector, APIManager


async def demonstrate_api_detection():
    """Demonstrate API key detection capabilities"""
    print("J.A.R.V.I.S. API Key Detection Demo")
    print("=" * 50)

    # Create detector
    detector = APIKeyDetector()

    # Test cases with different API key formats
    test_cases = [
        ("sk-test1234567890123456789012345678901234567890", "OpenAI format"),
        ("sk-ant-test123456789012345678901234567890", "Anthropic format"),
        ("AIzaSyTest1234567890abcdefghijklmnopqrstuvw", "Google API format"),
        ("test_cohere_key_12345678901234567890", "Generic key (no pattern match)"),
    ]

    print("\nTesting API Key Detection:")
    print("-" * 30)

    for test_key, description in test_cases:
        print(f"\nTesting: {description}")
        print(f"   Key: {test_key[:25]}...")

        try:
            provider, validation_info = await detector.detect_provider(test_key)

            if provider:
                print(f"   [+] Detected: {provider.value}")
                print(f"   [Confidence] {validation_info.get('confidence', 'N/A')}")
                print(f"   [Valid] {validation_info.get('valid', False)}")

                # Show capabilities if detected
                if validation_info.get('valid', False):
                    capabilities = await detector.detect_models_and_capabilities(provider, test_key)
                    models = capabilities.get('models', [])
                    features = capabilities.get('features', [])
                    print(f"   [Models] {', '.join(models[:3])}{'...' if len(models) > 3 else ''}")
                    print(f"   [Features] {', '.join(features)}")
            else:
                error = validation_info.get('error', 'Unknown error')
                print(f"   [-] Not detected: {error}")

        except Exception as e:
            print(f"   [Error] {str(e)}")

    print("\nSupported API Providers:")
    print("-" * 30)

    # Show all supported providers
    manager = APIManager(None)
    providers = manager.list_supported_providers()

    for provider in providers:
        configured = "[YES]" if provider.get("configured", False) else "[NO]"
        print(f"   {configured} {provider['display_name']}")
        print(f"      Formats: {', '.join(provider.get('key_formats', ['Various']))}")
        print(f"      Features: {', '.join(provider.get('supported_features', []))}")
        print()

    print("\nUsage Examples:")
    print("-" * 30)
    print("   * 'Add API key sk-1234567890abcdef'")
    print("   * 'Test API key sk-ant-abcdef123456'")
    print("   * 'List API providers'")
    print("   * 'Remove API provider openai'")

    print("\nThe system automatically detects the provider and configures it!")
    print("=" * 50)


async def demonstrate_real_api_test():
    """Demonstrate testing with a real API key format (without actual validation)"""
    print("\nReal API Key Testing Demo")
    print("=" * 40)

    detector = APIKeyDetector()

    # Example of what would happen with a real key
    print("This demo shows the detection logic without making actual API calls.")
    print("In real usage, the system would validate keys against provider APIs.")
    print()

    # Show pattern matching for different providers
    patterns = {
        "OpenAI": "sk-proj-1234567890abcdef...",
        "Anthropic": "sk-ant-1234567890abcdef...",
        "Google": "AIzaSy1234567890abcdef...",
        "Cohere": "Any string (validated via API)",
        "Stability AI": "sk-1234567890abcdef...",
        "ElevenLabs": "Any string (validated via API)"
    }

    print("API Key Patterns:")
    for provider, pattern in patterns.items():
        print(f"   {provider}: {pattern}")

    print("\nDetection Process:")
    print("   1. Pattern matching against known formats")
    print("   2. API validation against provider endpoints")
    print("   3. Capability detection (models, features, limits)")
    print("   4. Automatic configuration and storage")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_api_detection())
    asyncio.run(demonstrate_real_api_test())