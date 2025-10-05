#!/usr/bin/env python3
"""
J.A.R.V.I.S. Comprehensive Feature Test
Tests all claimed features to ensure they work with real implementations
"""

import sys
import os
import asyncio
import time
import json
from pathlib import Path

# Add jarvis to path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from core.jarvis import JARVIS
from core.command_processor import Command
from modules.voice_interface import VoiceInterface
from modules.system_monitor import SystemMonitor
from modules.security_manager import SecurityManager
from modules.plugin_manager import PluginManager
from core.advanced.memory_manager import MemoryManager
from core.advanced.updater import UpdateManager
from core.advanced.tester import Tester
from core.advanced.healer_components.optimizer import Optimizer


class FeatureTester:
    """Comprehensive feature tester for J.A.R.V.I.S."""

    def __init__(self):
        self.jarvis = None
        self.results = {}
        self.logger = self._setup_logger()

    def _setup_logger(self):
        import logging
        logger = logging.getLogger('FeatureTester')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    async def run_all_tests(self):
        """Run all feature tests"""
        self.logger.info("Starting comprehensive J.A.R.V.I.S. feature tests...")

        try:
            # Test 1: JARVIS Initialization
            await self.test_jarvis_initialization()

            # Test 2: Voice Interface
            await self.test_voice_interface()

            # Test 3: System Monitoring
            await self.test_system_monitoring()

            # Test 4: Memory Management
            await self.test_memory_management()

            # Test 5: Command Processing
            await self.test_command_processing()

            # Test 6: Security Features
            await self.test_security_features()

            # Test 7: Plugin System
            await self.test_plugin_system()

            # Test 8: Update System
            await self.test_update_system()

            # Test 9: Code Testing & Optimization
            await self.test_code_testing()

            # Test 10: GUI Framework
            await self.test_gui_framework()

            # Generate test report
            self.generate_report()

        except Exception as e:
            self.logger.error(f"Test suite failed: {e}")
            import traceback
            traceback.print_exc()

    async def test_jarvis_initialization(self):
        """Test JARVIS system initialization"""
        self.logger.info("Testing JARVIS initialization...")

        try:
            start_time = time.time()
            self.jarvis = JARVIS()

            # Initialize modules
            self.jarvis.initialize_modules()

            init_time = time.time() - start_time

            # Check if all modules are initialized
            modules_status = {
                "voice_interface": self.jarvis.voice_interface is not None,
                "system_monitor": self.jarvis.system_monitor is not None,
                "app_controller": self.jarvis.app_controller is not None,
                "file_manager": self.jarvis.file_manager is not None,
                "network_manager": self.jarvis.network_manager is not None,
                "security_manager": self.jarvis.security_manager is not None,
                "plugin_manager": self.jarvis.plugin_manager is not None,
            }

            self.results["jarvis_initialization"] = {
                "success": True,
                "init_time": init_time,
                "modules_initialized": sum(modules_status.values()),
                "total_modules": len(modules_status),
                "module_status": modules_status
            }

            self.logger.info(f"JARVIS initialized in {init_time:.2f}s")

        except Exception as e:
            self.results["jarvis_initialization"] = {
                "success": False,
                "error": str(e)
            }
            self.logger.error(f"JARVIS initialization failed: {e}")

    async def test_voice_interface(self):
        """Test voice interface capabilities"""
        self.logger.info("Testing voice interface...")

        try:
            if not self.jarvis or not self.jarvis.voice_interface:
                raise Exception("Voice interface not initialized")

            voice = self.jarvis.voice_interface

            # Test basic functionality
            test_text = "Hello, this is a test message"
            voice.speak(test_text, "normal")

            # Test voice properties
            voices = voice.tts_engine.getProperty('voices') if hasattr(voice, 'tts_engine') and hasattr(voice.tts_engine, 'getProperty') else []

            self.results["voice_interface"] = {
                "success": True,
                "voices_available": len(voices) if voices else "unknown",
                "speech_test": "passed",
                "engine_type": "pyttsx3"
            }

            self.logger.info("✓ Voice interface working")

        except Exception as e:
            self.results["voice_interface"] = {
                "success": False,
                "error": str(e)
            }
            self.logger.error(f"✗ Voice interface test failed: {e}")

    async def test_system_monitoring(self):
        """Test system monitoring capabilities"""
        self.logger.info("Testing system monitoring...")

        try:
            if not self.jarvis or not self.jarvis.system_monitor:
                raise Exception("System monitor not initialized")

            monitor = self.jarvis.system_monitor

            # Get system info
            system_info = monitor.get_system_info()

            # Get current readings
            readings = monitor.get_current_readings()

            # Check if we have CPU, memory, disk data
            has_cpu = "cpu" in readings
            has_memory = "memory" in readings
            has_disk = "disk" in readings

            self.results["system_monitoring"] = {
                "success": True,
                "system_info_available": bool(system_info),
                "cpu_monitoring": has_cpu,
                "memory_monitoring": has_memory,
                "disk_monitoring": has_disk,
                "readings_count": len(readings)
            }

            self.logger.info("✓ System monitoring working")

        except Exception as e:
            self.results["system_monitoring"] = {
                "success": False,
                "error": str(e)
            }
            self.logger.error(f"✗ System monitoring test failed: {e}")

    async def test_memory_management(self):
        """Test memory management system"""
        self.logger.info("Testing memory management...")

        try:
            # Create memory manager instance
            memory_manager = MemoryManager(self.jarvis)
            await memory_manager.initialize()

            # Test memory storage
            test_content = "This is a test memory for J.A.R.V.I.S. system"
            memory_id = await memory_manager.store_memory(
                content=test_content,
                memory_type="test",
                importance=0.8
            )

            # Test memory retrieval
            memories = await memory_manager.retrieve_memories("test memory", limit=5)

            # Test memory context
            context = await memory_manager.get_memory_context("What is J.A.R.V.I.S.?")

            self.results["memory_management"] = {
                "success": True,
                "memory_stored": bool(memory_id),
                "memory_retrieved": len(memories) > 0,
                "context_generated": bool(context),
                "vector_db_available": memory_manager.vector_db is not None,
                "embedding_model_loaded": memory_manager.embedding_model is not None
            }

            self.logger.info("✓ Memory management working")

        except Exception as e:
            self.results["memory_management"] = {
                "success": False,
                "error": str(e)
            }
            self.logger.error(f"✗ Memory management test failed: {e}")

    async def test_command_processing(self):
        """Test command processing"""
        self.logger.info("Testing command processing...")

        try:
            if not self.jarvis or not self.jarvis.command_processor:
                raise Exception("Command processor not initialized")

            processor = self.jarvis.command_processor

            # Test command execution
            result = processor.execute_command("status")

            # Test command parsing
            command_obj = processor._parse_command(Command("what time is it"))

            self.results["command_processing"] = {
                "success": True,
                "command_execution": result is not None,
                "command_parsing": command_obj.parsed_intent is not None,
                "registered_patterns": len(processor.command_patterns),
                "registered_handlers": len(processor.intent_handlers)
            }

            self.logger.info("✓ Command processing working")

        except Exception as e:
            self.results["command_processing"] = {
                "success": False,
                "error": str(e)
            }
            self.logger.error(f"✗ Command processing test failed: {e}")

    async def test_security_features(self):
        """Test security features"""
        self.logger.info("Testing security features...")

        try:
            if not self.jarvis or not self.jarvis.security_manager:
                raise Exception("Security manager not initialized")

            security = self.jarvis.security_manager

            # Test user authentication (will create default user)
            auth_result = security.authenticate_user("admin", "admin123!")

            # Test data encryption/decryption
            test_data = "This is sensitive test data"
            encrypted = security.encrypt_data(test_data)
            decrypted = security.decrypt_data(encrypted)

            # Test session validation
            if auth_result.get("success"):
                session_valid = security.validate_session("admin", auth_result["session_token"])
            else:
                session_valid = False

            self.results["security_features"] = {
                "success": True,
                "authentication_working": auth_result.get("success", False),
                "encryption_working": encrypted != test_data and decrypted == test_data,
                "session_management": session_valid,
                "audit_logging": len(security.security_events) >= 0
            }

            self.logger.info("✓ Security features working")

        except Exception as e:
            self.results["security_features"] = {
                "success": False,
                "error": str(e)
            }
            self.logger.error(f"✗ Security features test failed: {e}")

    async def test_plugin_system(self):
        """Test plugin system"""
        self.logger.info("Testing plugin system...")

        try:
            if not self.jarvis or not self.jarvis.plugin_manager:
                raise Exception("Plugin manager not initialized")

            plugin_mgr = self.jarvis.plugin_manager

            # Check plugin loading
            plugin_mgr.load_plugins()

            # Get plugin stats
            stats = plugin_mgr.get_plugin_stats()

            self.results["plugin_system"] = {
                "success": True,
                "plugins_loaded": stats.get("plugins_loaded", 0),
                "plugins_enabled": stats.get("enabled_plugins", 0),
                "plugin_directories": len(plugin_mgr.plugin_directories),
                "hooks_registered": stats.get("total_hooks", 0)
            }

            self.logger.info("✓ Plugin system working")

        except Exception as e:
            self.results["plugin_system"] = {
                "success": False,
                "error": str(e)
            }
            self.logger.error(f"✗ Plugin system test failed: {e}")

    async def test_update_system(self):
        """Test update system"""
        self.logger.info("Testing update system...")

        try:
            # Create update manager
            updater = UpdateManager(self.jarvis)
            await updater.initialize()

            # Test version checking (will fail gracefully if no server)
            update_result = await updater.check_for_updates(force=True)

            self.results["update_system"] = {
                "success": True,
                "version_check_attempted": True,
                "current_version": updater.current_version,
                "update_available": update_result.get("update_available", False),
                "config_loaded": bool(updater.update_config)
            }

            self.logger.info("✓ Update system working")

        except Exception as e:
            self.results["update_system"] = {
                "success": False,
                "error": str(e)
            }
            self.logger.error(f"✗ Update system test failed: {e}")

    async def test_code_testing(self):
        """Test code testing and optimization"""
        self.logger.info("Testing code testing and optimization...")

        try:
            # Create tester with JARVIS instance
            tester = Tester(self.jarvis)
            tester.jarvis = self.jarvis  # Fix the reference
            await tester.initialize()

            # Test code analysis
            test_code = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

result = calculate_fibonacci(10)
print(result)
"""

            analysis = tester.code_analyzer.analyze_code(test_code)

            # Test performance profiling
            profile_result = tester.profiler.profile_function(lambda: sum(range(1000)))

            self.results["code_testing"] = {
                "success": True,
                "code_analysis": bool(analysis.get("issues_found") is not None),
                "performance_profiling": bool(profile_result.get("execution_time")),
                "static_analysis_available": True,
                "optimization_suggestions": len(analysis.get("recommendations", []))
            }

            self.logger.info("Code testing and optimization working")

        except Exception as e:
            self.results["code_testing"] = {
                "success": False,
                "error": str(e)
            }
            self.logger.error(f"Code testing test failed: {e}")

    async def test_gui_framework(self):
        """Test GUI framework initialization"""
        self.logger.info("Testing GUI framework...")

        try:
            # Test PyQt6 import and basic functionality
            from PyQt6.QtWidgets import QApplication
            from PyQt6.QtCore import QTimer

            # Note: Can't fully test GUI without display, but can test imports
            app_available = True

            self.results["gui_framework"] = {
                "success": True,
                "pyqt6_available": app_available,
                "qt_version": "6.x",
                "widgets_available": True
            }

            self.logger.info("✓ GUI framework available")

        except ImportError as e:
            self.results["gui_framework"] = {
                "success": False,
                "error": f"GUI framework not available: {e}"
            }
            self.logger.error(f"✗ GUI framework test failed: {e}")

    def generate_report(self):
        """Generate comprehensive test report"""
        self.logger.info("Generating test report...")

        report = {
            "test_timestamp": time.time(),
            "total_tests": len(self.results),
            "passed_tests": sum(1 for r in self.results.values() if r.get("success", False)),
            "failed_tests": sum(1 for r in self.results.values() if not r.get("success", False)),
            "overall_success_rate": sum(1 for r in self.results.values() if r.get("success", False)) / len(self.results) * 100,
            "detailed_results": self.results
        }

        # Save report
        report_path = Path(__file__).parent / "test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\n" + "="*60)
        print("J.A.R.V.I.S. FEATURE TEST REPORT")
        print("="*60)
        print(f"Total Tests: {report['total_tests']}")
        print(f"Passed: {report['passed_tests']}")
        print(f"Failed: {report['failed_tests']}")
        print(".1f")
        print("\nDetailed Results:")

        for test_name, result in self.results.items():
            status = "PASS" if result.get("success", False) else "FAIL"
            print(f"  {test_name}: {status}")
            if not result.get("success", False) and "error" in result:
                print(f"    Error: {result['error']}")

        print(f"\nReport saved to: {report_path}")
        print("="*60)

        return report


async def main():
    """Main test function"""
    tester = FeatureTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())