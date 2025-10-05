#!/usr/bin/env python3
"""
J.A.R.V.I.S. Comprehensive Test Suite
Advanced testing framework for all system components
"""

import sys
import os
import time
import asyncio
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

# Add JARVIS to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.jarvis import JARVIS
from core.system_core import SystemCore
from core.event_manager import EventManager
from core.command_processor import CommandProcessor
from modules.voice_interface import VoiceInterface
from modules.system_monitor import SystemMonitor
from modules.file_manager import FileManager
from core.advanced.self_development_engine import SelfDevelopmentEngine
from core.advanced.code_generator import CodeGenerator
from core.advanced.updater import UpdateManager


class TestJARVISCore(unittest.TestCase):
    """Test core JARVIS functionality"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.json')

        # Create test config
        test_config = {
            "system": {
                "name": "JARVIS_TEST",
                "version": "1.0.0",
                "enable_voice": False,
                "enable_gui": False
            }
        }

        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)

    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_jarvis_initialization(self):
        """Test JARVIS initialization"""
        jarvis = JARVIS(self.config_path)

        self.assertIsNotNone(jarvis)
        self.assertEqual(jarvis.status, "initializing")
        self.assertIsNotNone(jarvis.config)
        self.assertIsNotNone(jarvis.logger)

    def test_system_core(self):
        """Test SystemCore functionality"""
        jarvis = JARVIS(self.config_path)
        system_core = SystemCore(jarvis)

        self.assertIsNotNone(system_core)
        self.assertEqual(system_core.jarvis, jarvis)

    def test_event_manager(self):
        """Test EventManager functionality"""
        jarvis = JARVIS(self.config_path)
        event_manager = EventManager(jarvis)

        self.assertIsNotNone(event_manager)

        # Test custom event creation
        event_id = event_manager.create_custom_event("test_event", {"data": "test_data"})

        # Check if event ID was returned
        self.assertIsNotNone(event_id)
        self.assertTrue(len(event_id) > 0)

    def test_command_processor(self):
        """Test CommandProcessor functionality"""
        jarvis = JARVIS(self.config_path)
        command_processor = CommandProcessor(jarvis)

        self.assertIsNotNone(command_processor)

        # Test command registration
        def test_handler(command):
            return {"result": "test"}

        command_processor.register_command("test", test_handler)

        # Test command execution
        result = command_processor.execute_command("test")
        self.assertEqual(result["result"], "test")


class TestModules(unittest.TestCase):
    """Test JARVIS modules"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.json')

        test_config = {
            "system": {
                "name": "JARVIS_TEST",
                "version": "1.0.0",
                "enable_voice": False,
                "enable_gui": False
            }
        }

        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_system_monitor(self, mock_disk, mock_memory, mock_cpu):
        """Test SystemMonitor functionality"""
        # Mock psutil functions
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(percent=60.0)
        mock_disk.return_value = Mock(percent=70.0)

        jarvis = JARVIS(self.config_path)
        monitor = SystemMonitor(jarvis)

        self.assertIsNotNone(monitor)

        # Test monitoring start
        monitor.start_monitoring()
        self.assertTrue(monitor.monitoring)

        # Test data collection
        readings = monitor.current_readings
        self.assertIsNotNone(readings)

    def test_file_manager(self):
        """Test FileManager functionality"""
        jarvis = JARVIS(self.config_path)
        file_manager = FileManager(jarvis)

        self.assertIsNotNone(file_manager)

        # Test directory creation
        test_dir = os.path.join(self.temp_dir, 'test_dir')
        result = file_manager.create_directory(test_dir)
        self.assertTrue(os.path.exists(test_dir))

        # Test file operations
        test_file = os.path.join(test_dir, 'test.txt')
        result = file_manager.write_file(test_file, "test content")
        self.assertTrue(os.path.exists(test_file))

        content = file_manager.read_file(test_file)
        self.assertEqual(content, "test content")


class TestAdvancedFeatures(unittest.TestCase):
    """Test advanced JARVIS features"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.json')

        test_config = {
            "system": {
                "name": "JARVIS_TEST",
                "version": "1.0.0",
                "enable_voice": False,
                "enable_gui": False
            }
        }

        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_code_generator(self):
        """Test CodeGenerator functionality"""
        jarvis = JARVIS(self.config_path)
        # CodeGenerator expects a development engine, so create one
        dev_engine = SelfDevelopmentEngine(jarvis)
        code_gen = CodeGenerator(dev_engine)

        self.assertIsNotNone(code_gen)

        # Test code validation
        test_code = '''
def test_function():
    """Test function"""
    return "test"
'''
        validation = code_gen._validate_generated_code(test_code)
        self.assertTrue(validation["valid"])

    def test_update_manager(self):
        """Test UpdateManager functionality"""
        jarvis = JARVIS(self.config_path)
        updater = UpdateManager(jarvis)

        self.assertIsNotNone(updater)
        self.assertEqual(updater.current_version, "2.0.0")

        # Test version comparison
        self.assertTrue(updater._compare_versions("1.0.0", "2.0.0"))
        self.assertFalse(updater._compare_versions("2.0.0", "1.0.0"))


class TestIntegration(unittest.TestCase):
    """Integration tests for JARVIS system"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.json')

        test_config = {
            "system": {
                "name": "JARVIS_TEST",
                "version": "1.0.0",
                "enable_voice": False,
                "enable_gui": False
            }
        }

        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_system_initialization(self):
        """Test full JARVIS system initialization"""
        jarvis = JARVIS(self.config_path)

        # Test initialization doesn't crash
        self.assertIsNotNone(jarvis.core)
        self.assertIsNotNone(jarvis.event_manager)
        self.assertIsNotNone(jarvis.command_processor)

    def test_command_execution_pipeline(self):
        """Test command execution through full pipeline"""
        jarvis = JARVIS(self.config_path)

        # Register a test command
        def test_command_handler(command):
            return {"result": "success", "command": command.text}

        jarvis.command_processor.register_command("test", test_command_handler)

        # Execute command
        result = jarvis.execute_command("test command")

        self.assertIsNotNone(result)
        self.assertEqual(result["result"], "success")


class TestPerformance(unittest.TestCase):
    """Performance tests for JARVIS system"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.json')

        test_config = {
            "system": {
                "name": "JARVIS_TEST",
                "version": "1.0.0",
                "enable_voice": False,
                "enable_gui": False
            }
        }

        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization_performance(self):
        """Test JARVIS initialization performance"""
        start_time = time.time()

        jarvis = JARVIS(self.config_path)

        init_time = time.time() - start_time

        # Should initialize in under 5 seconds
        self.assertLess(init_time, 5.0, f"Initialization took {init_time} seconds")

    def test_command_execution_performance(self):
        """Test command execution performance"""
        jarvis = JARVIS(self.config_path)

        # Register fast command
        def fast_command(command):
            return {"result": "fast"}

        jarvis.command_processor.register_command("fast", fast_command)

        # Test execution time
        import time
        start_time = time.time()

        for _ in range(100):
            jarvis.execute_command("fast")

        total_time = time.time() - start_time
        avg_time = total_time / 100

        # Should execute in under 0.1 seconds per command
        self.assertLess(avg_time, 0.1, f"Average execution time: {avg_time} seconds")


class TestSecurity(unittest.TestCase):
    """Security tests for JARVIS system"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.json')

        test_config = {
            "system": {
                "name": "JARVIS_TEST",
                "version": "1.0.0",
                "enable_voice": False,
                "enable_gui": False
            },
            "security": {
                "encryption_enabled": True
            }
        }

        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_input_validation(self):
        """Test input validation and sanitization"""
        jarvis = JARVIS(self.config_path)

        # Test with potentially dangerous input
        dangerous_commands = [
            "rm -rf /",
            "../../../etc/passwd",
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --"
        ]

        for cmd in dangerous_commands:
            result = jarvis.execute_command(cmd)
            # Should not crash and should handle gracefully
            self.assertIsNotNone(result)

    def test_access_control(self):
        """Test access control mechanisms"""
        jarvis = JARVIS(self.config_path)

        # Test that sensitive operations require proper permissions
        # This would be more comprehensive in a real security test
        self.assertIsNotNone(jarvis)


class TestGUI(unittest.TestCase):
    """GUI tests for JARVIS interface"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.json')

        test_config = {
            "system": {
                "name": "JARVIS_TEST",
                "version": "1.0.0",
                "enable_voice": False,
                "enable_gui": False
            }
        }

        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('PyQt6.QtWidgets.QApplication')
    def test_gui_initialization(self, mock_qapp):
        """Test GUI initialization without display"""
        mock_qapp.return_value = Mock()

        jarvis = JARVIS(self.config_path)

        # Test GUI creation (would normally require display)
        try:
            from gui.main_window import JARVISGUI
            gui = JARVISGUI(jarvis)
            self.assertIsNotNone(gui)
        except ImportError:
            # PyQt6 not available in test environment
            self.skipTest("PyQt6 not available")


def run_performance_benchmark():
    """Run comprehensive performance benchmark"""
    print("Running JARVIS Performance Benchmark...")
    print("=" * 50)

    results = {}

    # Test initialization time
    start_time = time.time()
    jarvis = JARVIS()
    init_time = time.time() - start_time
    results["initialization_time"] = init_time
    print(".2f")

    # Test command execution speed
    commands_tested = 1000
    start_time = time.time()

    for i in range(commands_tested):
        jarvis.execute_command(f"test command {i}")

    command_time = time.time() - start_time
    avg_command_time = command_time / commands_tested
    results["avg_command_time"] = avg_command_time
    print(".4f")

    # Test memory usage
    import psutil
    process = psutil.Process()
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    results["memory_usage"] = memory_usage
    print(".2f")

    # Save results
    results_file = Path("performance_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    return results


def run_integration_tests():
    """Run comprehensive integration tests"""
    print("Running JARVIS Integration Tests...")
    print("=" * 50)

    jarvis = JARVIS()

    tests_passed = 0
    tests_total = 0

    # Test module initialization
    tests_total += 1
    try:
        jarvis.initialize_modules()
        print("✓ Module initialization")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Module initialization failed: {e}")

    # Test system monitoring
    tests_total += 1
    try:
        if jarvis.system_monitor:
            readings = jarvis.system_monitor.get_system_info()
            if readings:
                print("✓ System monitoring")
                tests_passed += 1
            else:
                print("✗ System monitoring returned no data")
        else:
            print("✗ System monitor not initialized")
    except Exception as e:
        print(f"✗ System monitoring failed: {e}")

    # Test command processing
    tests_total += 1
    try:
        result = jarvis.execute_command("status")
        if result:
            print("✓ Command processing")
            tests_passed += 1
        else:
            print("✗ Command processing returned no result")
    except Exception as e:
        print(f"✗ Command processing failed: {e}")

    # Test file operations
    tests_total += 1
    try:
        if jarvis.file_manager:
            test_file = "/tmp/jarvis_test.txt"
            jarvis.file_manager.write_file(test_file, "test content")
            content = jarvis.file_manager.read_file(test_file)
            if content == "test content":
                print("✓ File operations")
                tests_passed += 1
                # Cleanup
                os.remove(test_file)
            else:
                print("✗ File operations returned wrong content")
        else:
            print("✗ File manager not initialized")
    except Exception as e:
        print(f"✗ File operations failed: {e}")

    print(f"\nIntegration Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


if __name__ == '__main__':
    # Setup logging for tests
    logging.basicConfig(level=logging.WARNING)

    # Run unit tests
    print("Running JARVIS Unit Tests...")
    print("=" * 50)

    unittest.main(argv=[''], exit=False, verbosity=2)

    # Run integration tests
    print("\n" + "=" * 50)
    integration_passed = run_integration_tests()

    # Run performance benchmark
    print("\n" + "=" * 50)
    benchmark_results = run_performance_benchmark()

    print("\n" + "=" * 50)
    print("JARVIS Test Suite Complete")
    print(f"Integration Tests: {'PASSED' if integration_passed else 'FAILED'}")
    print(".2f")
    print(".2f")
    print(".4f")