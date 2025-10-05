"""
J.A.R.V.I.S. Code Generator
AI-powered code generation for autonomous development
"""

import os
import time
import asyncio
import ast
import json
from typing import Dict, List, Optional, Any
import logging


class CodeGenerator:
    """
    Advanced code generation system
    Uses AI APIs to generate high-quality, working Python code
    """

    def __init__(self, development_engine):
        """
        Initialize code generator

        Args:
            development_engine: Reference to self-development engine
        """
        self.engine = development_engine
        self.jarvis = development_engine.jarvis
        self.logger = logging.getLogger('JARVIS.CodeGenerator')

        # Code generation settings
        self.generation_config = {
            "temperature": 0.7,
            "max_tokens": 2000,
            "model_preferences": ["gpt-4", "claude-3-opus", "gemini-pro"],
            "code_style": "pep8",
            "include_tests": True,
            "include_docstrings": True,
            "error_handling": "comprehensive"
        }

        # Generated code tracking
        self.generated_files = []
        self.code_quality_scores = {}

    async def initialize(self):
        """Initialize code generator"""
        try:
            self.logger.info("Initializing code generator...")

            # Test code generation capability
            await self._test_generation()

            self.logger.info("Code generator initialized")

        except Exception as e:
            self.logger.error(f"Error initializing code generator: {e}")
            raise

    async def _test_generation(self):
        """Test code generation capability"""
        try:
            test_prompt = "Generate a simple Python function that adds two numbers."

            # Use API manager to generate test code
            if hasattr(self.jarvis, 'api_manager'):
                request = self.jarvis.api_manager.APIRequest(
                    provider=self.jarvis.api_manager.APIProvider.OPENAI,
                    model="gpt-3.5-turbo",
                    prompt=test_prompt,
                    request_type="text"
                )

                response = await self.jarvis.api_manager.make_request(request)

                if response.success:
                    self.logger.info("✓ Code generation test successful")
                else:
                    self.logger.warning("✗ Code generation test failed")

        except Exception as e:
            self.logger.error(f"Error testing code generation: {e}")

    async def generate_code(self,
                           task_description: str,
                           reasoning_data: Dict[str, Any],
                           requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate code for a development task

        Args:
            task_description: Description of what to build
            reasoning_data: Research and reasoning results
            requirements: Specific requirements

        Returns:
            Code generation result
        """
        try:
            self.logger.info(f"Generating code for: {task_description}")

            # Build comprehensive prompt
            prompt = await self._build_generation_prompt(task_description, reasoning_data, requirements)

            # Generate code using AI API
            code_result = await self._generate_with_ai(prompt)

            if not code_result.get("success", False):
                return code_result

            # Validate generated code
            validation_result = self._validate_generated_code(code_result["code"])

            if not validation_result["valid"]:
                # Try to fix code
                fixed_code = await self._fix_generated_code(code_result["code"], validation_result["errors"])
                if fixed_code:
                    code_result["code"] = fixed_code
                    validation_result = self._validate_generated_code(fixed_code)

            # Generate tests if requested
            if self.generation_config["include_tests"]:
                test_code = await self._generate_tests(code_result["code"], task_description)
                code_result["tests"] = test_code

            # Calculate quality score
            quality_score = self._calculate_code_quality(code_result["code"])
            code_result["quality_score"] = quality_score

            self.logger.info(f"Code generation completed with quality score: {quality_score}")

            return code_result

        except Exception as e:
            self.logger.error(f"Error generating code: {e}")
            return {
                "success": False,
                "error": str(e),
                "code": ""
            }

    async def _build_generation_prompt(self,
                                      task_description: str,
                                      reasoning_data: Dict[str, Any],
                                      requirements: Dict[str, Any]) -> str:
        """Build comprehensive prompt for code generation"""
        try:
            # Extract key information from reasoning data
            implementation_plan = reasoning_data.get("implementation_plan", "")
            technical_requirements = reasoning_data.get("technical_requirements", [])
            dependencies = reasoning_data.get("dependencies", [])

            # Build structured prompt
            prompt = f"""
You are an expert Python developer creating code for J.A.R.V.I.S., an advanced AI personal assistant.

TASK DESCRIPTION:
{task_description}

IMPLEMENTATION REQUIREMENTS:
{json.dumps(requirements, indent=2)}

IMPLEMENTATION PLAN:
{implementation_plan}

TECHNICAL REQUIREMENTS:
{chr(10).join(f"- {req}" for req in technical_requirements)}

DEPENDENCIES:
{chr(10).join(f"- {dep}" for dep in dependencies)}

CODE REQUIREMENTS:
1. Follow PEP 8 style guidelines
2. Include comprehensive docstrings
3. Implement proper error handling
4. Use type hints where appropriate
5. Make code modular and reusable
6. Include logging for debugging
7. Handle edge cases gracefully
8. Follow security best practices

Generate complete, working Python code that implements the requested feature. Include all necessary imports, class definitions, methods, and helper functions.

Return only the Python code without markdown formatting or explanations.
"""

            return prompt

        except Exception as e:
            self.logger.error(f"Error building generation prompt: {e}")
            return task_description

    async def _generate_with_ai(self, prompt: str) -> Dict[str, Any]:
        """Generate code using AI API"""
        try:
            # Use API manager to generate code
            if hasattr(self.jarvis, 'api_manager'):
                request = self.jarvis.api_manager.APIRequest(
                    provider=self.jarvis.api_manager.APIProvider.OPENAI,
                    model="gpt-4",
                    prompt=prompt,
                    request_type="text",
                    parameters={
                        "temperature": self.generation_config["temperature"],
                        "max_tokens": self.generation_config["max_tokens"]
                    }
                )

                response = await self.jarvis.api_manager.make_request(request)

                if response.success:
                    return {
                        "success": True,
                        "code": response.response,
                        "tokens_used": response.tokens_used,
                        "cost": response.cost
                    }
                else:
                    return {
                        "success": False,
                        "error": response.error,
                        "code": ""
                    }

            # Fallback to simple template
            return {
                "success": True,
                "code": self._generate_fallback_code(prompt),
                "tokens_used": 0,
                "cost": 0
            }

        except Exception as e:
            self.logger.error(f"Error generating with AI: {e}")
            return {
                "success": False,
                "error": str(e),
                "code": ""
            }

    def _generate_fallback_code(self, prompt: str) -> str:
        """Generate fallback code when AI is unavailable"""
        return f'''
"""
Generated code based on: {prompt[:100]}...
"""

def generated_function():
    """Generated function"""
    return "Code generation fallback"
'''

    def _validate_generated_code(self, code: str) -> Dict[str, Any]:
        """Validate generated code"""
        try:
            errors = []
            warnings = []

            # Check if code is valid Python
            try:
                ast.parse(code)
            except SyntaxError as e:
                errors.append(f"Syntax error: {e}")
            except Exception as e:
                errors.append(f"Parse error: {e}")

            # Check for required elements
            if "def " not in code and "class " not in code:
                warnings.append("No functions or classes found")

            # Check for docstrings
            if self.generation_config["include_docstrings"]:
                lines = code.split('\n')
                has_docstring = False
                for i, line in enumerate(lines):
                    if 'def ' in line or 'class ' in line:
                        # Check next few lines for docstring
                        for j in range(i + 1, min(i + 4, len(lines))):
                            if '"""' in lines[j] or "'''" in lines[j]:
                                has_docstring = True
                                break
                        break

                if not has_docstring:
                    warnings.append("Missing docstrings")

            # Check for error handling
            if self.generation_config["error_handling"] == "comprehensive":
                if "try:" not in code and "except" not in code:
                    warnings.append("No error handling found")

            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "syntax_valid": len(errors) == 0
            }

        except Exception as e:
            return {
                "valid": False,
                "errors": [str(e)],
                "warnings": [],
                "syntax_valid": False
            }

    async def _fix_generated_code(self, code: str, errors: List[str]) -> Optional[str]:
        """Attempt to fix generated code"""
        try:
            # Use AI to fix code
            fix_prompt = f"""
The following Python code has errors. Please fix them:

CODE:
{code}

ERRORS:
{chr(10).join(errors)}

Please provide the corrected code only.
"""

            fix_request = self.jarvis.api_manager.APIRequest(
                provider=self.jarvis.api_manager.APIProvider.OPENAI,
                model="gpt-4",
                prompt=fix_prompt,
                request_type="text"
            )

            fix_response = await self.jarvis.api_manager.make_request(fix_request)

            if fix_response.success:
                return fix_response.response

        except Exception as e:
            self.logger.error(f"Error fixing code: {e}")

        return None

    async def _generate_tests(self, code: str, task_description: str) -> str:
        """Generate tests for the code"""
        try:
            test_prompt = f"""
Generate comprehensive unit tests for the following Python code:

CODE:
{code}

TASK: {task_description}

Generate tests using pytest framework that cover:
1. Normal functionality
2. Edge cases
3. Error conditions
4. Input validation

Include test file structure and all necessary imports.
"""

            test_request = self.jarvis.api_manager.APIRequest(
                provider=self.jarvis.api_manager.APIProvider.OPENAI,
                model="gpt-4",
                prompt=test_prompt,
                request_type="text"
            )

            test_response = await self.jarvis.api_manager.make_request(test_request)

            if test_response.success:
                return test_response.response

        except Exception as e:
            self.logger.error(f"Error generating tests: {e}")

        return "# Tests would be generated here"

    def _calculate_code_quality(self, code: str) -> float:
        """Calculate code quality score"""
        try:
            score = 100.0

            # Check for docstrings
            if '"""' not in code:
                score -= 10

            # Check for error handling
            if "try:" not in code and "except" not in code:
                score -= 15

            # Check for type hints
            if "def " in code and "->" not in code:
                score -= 5

            # Check for logging
            if "logger" not in code.lower():
                score -= 5

            # Check for proper imports
            lines = code.split('\n')
            has_imports = any("import " in line for line in lines)
            if not has_imports:
                score -= 10

            return max(0, score)

        except Exception as e:
            self.logger.error(f"Error calculating code quality: {e}")
            return 50.0

    async def improve_existing_code(self,
                                   file_path: str,
                                   improvement_type: str = "performance") -> Dict[str, Any]:
        """
        Improve existing code

        Args:
            file_path: Path to file to improve
            improvement_type: Type of improvement (performance, security, readability)

        Returns:
            Improvement result
        """
        try:
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "error": f"File not found: {file_path}"
                }

            with open(file_path, 'r') as f:
                original_code = f.read()

            # Generate improvement prompt
            improvement_prompt = f"""
Improve the following Python code for {improvement_type}:

ORIGINAL CODE:
{original_code}

Focus on:
1. {improvement_type.title()} improvements
2. Code readability and maintainability
3. Best practices and modern Python features
4. Error handling and edge cases

Provide the improved code only.
"""

            improvement_request = self.jarvis.api_manager.APIRequest(
                provider=self.jarvis.api_manager.APIProvider.OPENAI,
                model="gpt-4",
                prompt=improvement_prompt,
                request_type="text"
            )

            response = await self.jarvis.api_manager.make_request(improvement_request)

            if response.success:
                return {
                    "success": True,
                    "original_code": original_code,
                    "improved_code": response.response,
                    "improvement_type": improvement_type
                }
            else:
                return {
                    "success": False,
                    "error": response.error
                }

        except Exception as e:
            self.logger.error(f"Error improving code: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def generate_plugin_code(self,
                                  plugin_name: str,
                                  plugin_description: str,
                                  plugin_features: List[str]) -> Dict[str, Any]:
        """Generate complete plugin code"""
        try:
            plugin_prompt = f"""
Generate a complete J.A.R.V.I.S. plugin with the following specifications:

PLUGIN NAME: {plugin_name}
DESCRIPTION: {plugin_description}
FEATURES: {', '.join(plugin_features)}

Generate:
1. Plugin class inheriting from Plugin
2. Command handlers for each feature
3. Proper initialization and cleanup
4. Configuration management
5. Error handling
6. Documentation

Follow J.A.R.V.I.S. plugin architecture and coding standards.
"""

            request = self.jarvis.api_manager.APIRequest(
                provider=self.jarvis.api_manager.APIProvider.OPENAI,
                model="gpt-4",
                prompt=plugin_prompt,
                request_type="text"
            )

            response = await self.jarvis.api_manager.make_request(request)

            if response.success:
                return {
                    "success": True,
                    "plugin_code": response.response,
                    "plugin_name": plugin_name
                }
            else:
                return {
                    "success": False,
                    "error": response.error
                }

        except Exception as e:
            self.logger.error(f"Error generating plugin code: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def shutdown(self):
        """Shutdown code generator"""
        try:
            self.logger.info("Shutting down code generator...")

            # Save generated files list
            self._save_generated_files()

            self.logger.info("Code generator shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down code generator: {e}")

    def _save_generated_files(self):
        """Save list of generated files"""
        try:
            files_data = {
                "generated_files": self.generated_files,
                "quality_scores": self.code_quality_scores,
                "last_updated": time.time()
            }

            files_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'generated_files.json')

            os.makedirs(os.path.dirname(files_path), exist_ok=True)
            with open(files_path, 'w') as f:
                json.dump(files_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving generated files: {e}")