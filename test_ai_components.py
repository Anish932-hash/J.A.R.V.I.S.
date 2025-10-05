#!/usr/bin/env python3
"""
J.A.R.V.I.S. AI Components Test
Test self-development engine and application healer functionality
"""

import sys
import os
import asyncio
import time
from pathlib import Path

# Add jarvis to path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from core.jarvis import JARVIS
from core.advanced.self_development_engine import SelfDevelopmentEngine
from core.advanced.application_healer import ApplicationHealer


class AIComponentsTester:
    """Test AI components functionality"""

    def __init__(self):
        self.jarvis = None
        self.results = {}
        self.logger = self._setup_logger()

    def _setup_logger(self):
        import logging
        logger = logging.getLogger('AIComponentsTester')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    async def test_ai_components(self):
        """Test AI components"""
        self.logger.info("Testing J.A.R.V.I.S. AI Components...")

        try:
            # Initialize JARVIS
            self.jarvis = JARVIS()
            self.jarvis.initialize_modules()

            # Test Self-Development Engine
            await self.test_self_development_engine()

            # Test Application Healer
            await self.test_application_healer()

            # Generate report
            self.generate_report()

        except Exception as e:
            self.logger.error(f"AI Components test failed: {e}")
            import traceback
            traceback.print_exc()

    async def test_self_development_engine(self):
        """Test self-development engine"""
        self.logger.info("Testing Self-Development Engine...")

        try:
            # Create self-development engine
            sde = SelfDevelopmentEngine(self.jarvis)
            await sde.initialize()

            # Test task creation
            task_id = await sde.create_task(
                task_type="feature",
                description="Add user profile management feature",
                priority=5
            )

            # Test code generation
            code_request = {
                "task": "Create a user authentication function",
                "requirements": ["Use secure password hashing", "Return user object"],
                "language": "python"
            }

            generated_code = await sde.code_generator.generate_code(code_request)

            # Test reasoning
            reasoning_result = await sde.reasoning_engine.reason(
                task_description="How to implement secure user authentication?",
                research_data=[],
                requirements={"security": "high", "scalability": "medium"}
            )

            self.results["self_development_engine"] = {
                "success": True,
                "task_creation": bool(task_id),
                "code_generation": bool(generated_code.get("code")),
                "reasoning_engine": bool(reasoning_result.get("plan")),
                "validator_available": hasattr(sde, 'validator'),
                "web_searcher_available": hasattr(sde, 'web_searcher')
            }

            self.logger.info("Self-Development Engine working")

        except Exception as e:
            self.results["self_development_engine"] = {
                "success": False,
                "error": str(e)
            }
            self.logger.error(f"Self-Development Engine test failed: {e}")

    async def test_application_healer(self):
        """Test application healer"""
        self.logger.info("Testing Application Healer...")

        try:
            # Create application healer
            healer = ApplicationHealer(self.jarvis)
            await healer.initialize()

            # Test health check
            health_status = await healer.check_application_health()

            # Test error detection
            test_error = Exception("Test application error")
            error_analysis = healer.error_detector.analyze_error(test_error)

            # Test fix generation
            fix_suggestion = healer.fix_generator.generate_fix(
                error_info={"type": "AttributeError", "message": "'NoneType' object has no attribute 'method'"},
                context={"file": "test.py", "line": 42}
            )

            # Test recovery
            recovery_plan = healer.recovery_manager.create_recovery_plan(
                application_name="test_app",
                failure_type="crash",
                context={"pid": 1234, "error_count": 3}
            )

            self.results["application_healer"] = {
                "success": True,
                "health_check": bool(health_status),
                "error_detection": bool(error_analysis),
                "fix_generation": bool(fix_suggestion),
                "recovery_planning": bool(recovery_plan),
                "optimizer_available": hasattr(healer, 'optimizer'),
                "predictor_available": hasattr(healer, 'predictor')
            }

            self.logger.info("Application Healer working")

        except Exception as e:
            self.results["application_healer"] = {
                "success": False,
                "error": str(e)
            }
            self.logger.error(f"Application Healer test failed: {e}")

    def generate_report(self):
        """Generate test report"""
        self.logger.info("Generating AI Components test report...")

        report = {
            "test_timestamp": time.time(),
            "total_tests": len(self.results),
            "passed_tests": sum(1 for r in self.results.values() if r.get("success", False)),
            "failed_tests": sum(1 for r in self.results.values() if not r.get("success", False)),
            "overall_success_rate": sum(1 for r in self.results.values() if r.get("success", False)) / len(self.results) * 100,
            "detailed_results": self.results
        }

        # Save report
        report_path = Path(__file__).parent / "ai_components_test_report.json"
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\n" + "="*60)
        print("J.A.R.V.I.S. AI COMPONENTS TEST REPORT")
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
    tester = AIComponentsTester()
    await tester.test_ai_components()


if __name__ == "__main__":
    asyncio.run(main())