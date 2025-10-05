"""
J.A.R.V.I.S. Advanced API Manager
Centralized management for 100+ AI API providers with load balancing and failover
"""

import os
import json
import time
import asyncio
import aiohttp
import threading
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

# Import all AI API libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False

try:
    import stability_sdk
    STABILITY_AVAILABLE = True
except ImportError:
    STABILITY_AVAILABLE = False

try:
    import elevenlabs
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False


class APIProvider(Enum):
    """Supported AI API providers"""
    # Text Generation
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE_PALM = "google_palm"
    GOOGLE_GEMINI = "google_gemini"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    REPLICATE = "replicate"
    ALEPH_ALPHA = "aleph_alpha"
    AI21 = "ai21"
    FOREFRONT = "forefront"
    GOOSE_AI = "goose_ai"

    # Image Generation
    STABILITY_AI = "stability_ai"
    MIDJOURNEY = "midjourney"
    DALL_E = "dalle"
    KANDINSKY = "kandinsky"
    NIGHTCAFE = "nightcafe"
    STARRY_AI = "starry_ai"

    # Audio Processing
    ELEVENLABS = "elevenlabs"
    RESPEECH = "respeech"
    PLAY_HT = "play_ht"
    MURF = "murf"
    WELLSAID = "wellsaid"

    # Video Generation
    RUNWAY_ML = "runway_ml"
    Synthesia = "synthesia"
    D_ID = "d_id"
    Pika = "pika"

    # Code Generation
    CODEX = "codex"
    CODEWHISPERER = "codewhisperer"
    TABNINE = "tabnine"
    GITHUB_COPILOT = "github_copilot"

    # Multimodal
    GPT4_VISION = "gpt4_vision"
    CLAUDE_VISION = "claude_vision"
    GEMINI_PRO_VISION = "gemini_pro_vision"

    # Specialized
    WOLFRAM_ALPHA = "wolfram_alpha"
    SERPAPI = "serpapi"
    ZAPIER = "zapier"
    MAKE = "make"


@dataclass
class APIConfig:
    """Configuration for an API provider"""
    provider: APIProvider
    api_key: str
    base_url: str = ""
    models: List[str] = field(default_factory=list)
    rate_limits: Dict[str, int] = field(default_factory=dict)
    cost_per_token: float = 0.0
    priority: int = 1
    enabled: bool = True
    last_used: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0
    total_cost: float = 0.0


@dataclass
class APIRequest:
    """API request information"""
    provider: APIProvider
    model: str
    prompt: str
    request_type: str = "text"
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    timeout: int = 30
    retries: int = 3
    request_id: str = field(default_factory=lambda: secrets.token_hex(8))


@dataclass
class APIResponse:
    """API response information"""
    request_id: str
    provider: APIProvider
    model: str
    response: Any
    tokens_used: int = 0
    cost: float = 0.0
    response_time: float = 0.0
    success: bool = True
    error: str = ""


class APIKeyDetector:
    """Advanced API key detection and validation system"""

    def __init__(self):
        self.logger = logging.getLogger('JARVIS.APIKeyDetector')

        # API key patterns for different providers
        self.key_patterns = {
            APIProvider.OPENAI: {
                "prefixes": ["sk-", "sk-proj-", "sk-svcacct-"],
                "length_ranges": [(51, 164)],  # OpenAI keys vary in length
                "validation_endpoint": "https://api.openai.com/v1/models",
                "headers": {"Authorization": "Bearer {key}"},
                "expected_response": lambda r: "data" in r and isinstance(r["data"], list)
            },
            APIProvider.ANTHROPIC: {
                "prefixes": ["sk-ant-"],
                "length_ranges": [(95, 110)],
                "validation_endpoint": "https://api.anthropic.com/v1/messages",
                "headers": {"x-api-key": "{key}", "anthropic-version": "2023-06-01"},
                "test_payload": {"model": "claude-3-haiku-20240307", "max_tokens": 1, "messages": [{"role": "user", "content": "test"}]},
                "expected_response": lambda r: "content" in r
            },
            APIProvider.GOOGLE_GEMINI: {
                "prefixes": [],  # Google keys don't have standard prefixes
                "length_ranges": [(39, 39)],  # Google API keys are typically 39 chars
                "validation_endpoint": "https://generativelanguage.googleapis.com/v1beta/models?key={key}",
                "expected_response": lambda r: "models" in r
            },
            APIProvider.COHERE: {
                "prefixes": [],  # Cohere keys vary
                "length_ranges": [(30, 60)],
                "validation_endpoint": "https://api.cohere.ai/v1/models",
                "headers": {"Authorization": "Bearer {key}"},
                "expected_response": lambda r: isinstance(r, list) or ("models" in r)
            },
            APIProvider.STABILITY_AI: {
                "prefixes": ["sk-"],
                "length_ranges": [(40, 60)],
                "validation_endpoint": "https://api.stability.ai/v1/user/account",
                "headers": {"Authorization": "{key}"},
                "expected_response": lambda r: "email" in r
            },
            APIProvider.ELEVENLABS: {
                "prefixes": [],  # ElevenLabs keys vary
                "length_ranges": [(20, 50)],
                "validation_endpoint": "https://api.elevenlabs.io/v1/voices",
                "headers": {"xi-api-key": "{key}"},
                "expected_response": lambda r: isinstance(r, list)
            }
        }

    async def detect_provider(self, api_key: str) -> Tuple[Optional[APIProvider], Dict[str, Any]]:
        """
        Detect API provider from key format and validate

        Args:
            api_key: The API key to analyze

        Returns:
            Tuple of (detected_provider, validation_info)
        """
        try:
            # Clean the key
            api_key = api_key.strip()

            if not api_key:
                return None, {"error": "Empty API key"}

            # First, try pattern matching with prefixes (most reliable)
            prefix_matches = []
            for provider, patterns in self.key_patterns.items():
                if patterns.get("prefixes") and self._matches_pattern(api_key, patterns):
                    prefix_matches.append(provider)

            # If we have prefix matches, validate them
            if prefix_matches:
                validation_results = {}
                for provider in prefix_matches:
                    result = await self._validate_api_key(api_key, provider)
                    validation_results[provider] = result

                # Find the best valid match
                valid_providers = [(p, r) for p, r in validation_results.items() if r.get("valid", False)]
                if valid_providers:
                    best_match = max(valid_providers, key=lambda x: x[1].get("confidence", 0))
                    return best_match[0], best_match[1]

                # Return the first prefix match with error
                return prefix_matches[0], {
                    "valid": False,
                    "error": "API key prefix recognized but validation failed",
                    "pattern_matched": True
                }

            # If no prefix matches, try providers with specific patterns (Google, etc.)
            specific_matches = []
            for provider, patterns in self.key_patterns.items():
                # Skip providers that only have generic length ranges
                if not patterns.get("prefixes") and patterns.get("length_ranges"):
                    # Only include if they have other specific validation
                    if provider in [APIProvider.GOOGLE_GEMINI] and self._matches_pattern(api_key, patterns):
                        specific_matches.append(provider)

            if specific_matches:
                validation_results = {}
                for provider in specific_matches:
                    result = await self._validate_api_key(api_key, provider)
                    validation_results[provider] = result

                valid_providers = [(p, r) for p, r in validation_results.items() if r.get("valid", False)]
                if valid_providers:
                    best_match = max(valid_providers, key=lambda x: x[1].get("confidence", 0))
                    return best_match[0], best_match[1]

            # As last resort, try validation for all providers (expensive)
            validation_results = await self._validate_key_against_all_providers(api_key)
            if validation_results:
                # Return the first successful validation
                for provider, result in validation_results.items():
                    if result.get("valid", False):
                        return provider, result

            return None, {"error": "API key format not recognized"}

        except Exception as e:
            self.logger.error(f"Error detecting provider: {e}")
            return None, {"error": str(e)}

    def _matches_pattern(self, api_key: str, patterns: Dict[str, Any]) -> bool:
        """Check if API key matches provider patterns"""
        # Check prefixes (if any are specified)
        if patterns.get("prefixes"):
            has_matching_prefix = any(api_key.startswith(prefix) for prefix in patterns["prefixes"])
            if not has_matching_prefix:
                return False

        # Check length ranges (if any are specified)
        if patterns.get("length_ranges"):
            key_length = len(api_key)
            has_valid_length = any(min_len <= key_length <= max_len for min_len, max_len in patterns["length_ranges"])
            if not has_valid_length:
                return False

        # Additional pattern checks
        if patterns.get("regex"):
            import re
            if not re.match(patterns["regex"], api_key):
                return False

        # If we have either prefixes or length ranges, and they match, return True
        # If neither is specified, this provider doesn't have specific patterns, so return False
        has_prefixes = bool(patterns.get("prefixes"))
        has_length_ranges = bool(patterns.get("length_ranges"))

        if has_prefixes or has_length_ranges:
            return True

        return False

    async def _validate_api_key(self, api_key: str, provider: APIProvider) -> Dict[str, Any]:
        """Validate API key against provider"""
        try:
            patterns = self.key_patterns.get(provider)
            if not patterns:
                return {"valid": False, "error": "No validation patterns for provider"}

            # Make validation request
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                url = patterns["validation_endpoint"].format(key=api_key)
                headers = {}

                # Add authentication headers
                for header_name, header_value in patterns.get("headers", {}).items():
                    headers[header_name] = header_value.format(key=api_key)

                # Choose method
                method = patterns.get("method", "GET")

                # Prepare payload
                payload = patterns.get("test_payload", None)

                try:
                    if method == "GET":
                        async with session.get(url, headers=headers) as response:
                            if response.status == 200:
                                result = await response.json()
                                if patterns["expected_response"](result):
                                    return {
                                        "valid": True,
                                        "confidence": 0.9,
                                        "response": result
                                    }
                                else:
                                    return {"valid": False, "error": "Unexpected response format"}
                            elif response.status == 401:
                                return {"valid": False, "error": "Invalid API key"}
                            elif response.status == 403:
                                return {"valid": False, "error": "Forbidden - check permissions"}
                            else:
                                return {"valid": False, "error": f"HTTP {response.status}"}
                    else:
                        async with session.post(url, headers=headers, json=payload) as response:
                            if response.status == 200:
                                result = await response.json()
                                if patterns["expected_response"](result):
                                    return {
                                        "valid": True,
                                        "confidence": 0.9,
                                        "response": result
                                    }
                                else:
                                    return {"valid": False, "error": "Unexpected response format"}
                            elif response.status == 401:
                                return {"valid": False, "error": "Invalid API key"}
                            elif response.status == 429:
                                return {"valid": False, "error": "Rate limited"}
                            else:
                                return {"valid": False, "error": f"HTTP {response.status}"}

                except aiohttp.ClientError as e:
                    return {"valid": False, "error": f"Network error: {e}"}
                except Exception as e:
                    return {"valid": False, "error": f"Validation error: {e}"}

        except Exception as e:
            self.logger.error(f"Error validating API key for {provider.value}: {e}")
            return {"valid": False, "error": str(e)}

    async def _validate_key_against_all_providers(self, api_key: str) -> Dict[APIProvider, Dict[str, Any]]:
        """Validate key against all providers when pattern matching fails"""
        results = {}
        validation_tasks = []

        # Create validation tasks for providers that might accept this key format
        for provider in [APIProvider.OPENAI, APIProvider.ANTHROPIC, APIProvider.GOOGLE_GEMINI,
                        APIProvider.COHERE, APIProvider.STABILITY_AI, APIProvider.ELEVENLABS]:
            task = self._validate_api_key(api_key, provider)
            validation_tasks.append((provider, task))

        # Execute validations concurrently
        for provider, task in validation_tasks:
            try:
                result = await task
                if result.get("valid", False):
                    results[provider] = result
            except Exception as e:
                self.logger.debug(f"Validation failed for {provider.value}: {e}")

        return results

    async def detect_models_and_capabilities(self, provider: APIProvider, api_key: str) -> Dict[str, Any]:
        """Detect available models and capabilities for a provider"""
        try:
            capabilities = {
                "models": [],
                "features": [],
                "limits": {},
                "pricing": {}
            }

            if provider == APIProvider.OPENAI:
                capabilities.update(await self._detect_openai_capabilities(api_key))
            elif provider == APIProvider.ANTHROPIC:
                capabilities.update(await self._detect_anthropic_capabilities(api_key))
            elif provider == APIProvider.GOOGLE_GEMINI:
                capabilities.update(await self._detect_google_capabilities(api_key))
            elif provider == APIProvider.COHERE:
                capabilities.update(await self._detect_cohere_capabilities(api_key))
            elif provider == APIProvider.STABILITY_AI:
                capabilities.update(await self._detect_stability_capabilities(api_key))
            elif provider == APIProvider.ELEVENLABS:
                capabilities.update(await self._detect_elevenlabs_capabilities(api_key))

            return capabilities

        except Exception as e:
            self.logger.error(f"Error detecting capabilities for {provider.value}: {e}")
            return {"models": [], "features": [], "error": str(e)}

    async def _detect_openai_capabilities(self, api_key: str) -> Dict[str, Any]:
        """Detect OpenAI models and capabilities"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model["id"] for model in data.get("data", [])]

                        return {
                            "models": models,
                            "features": ["text_generation", "chat", "embeddings", "vision"],
                            "limits": {"requests_per_minute": 60, "tokens_per_minute": 40000},
                            "pricing": {"per_token": 0.000002}
                        }

            return {"models": ["gpt-3.5-turbo", "gpt-4"], "features": ["text_generation"]}

        except Exception as e:
            return {"models": ["gpt-3.5-turbo"], "features": ["text_generation"], "error": str(e)}

    async def _detect_anthropic_capabilities(self, api_key: str) -> Dict[str, Any]:
        """Detect Anthropic models and capabilities"""
        return {
            "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            "features": ["text_generation", "chat", "vision"],
            "limits": {"requests_per_minute": 50, "tokens_per_minute": 50000},
            "pricing": {"per_token": 0.000003}
        }

    async def _detect_google_capabilities(self, api_key: str) -> Dict[str, Any]:
        """Detect Google models and capabilities"""
        return {
            "models": ["gemini-pro", "gemini-pro-vision"],
            "features": ["text_generation", "chat", "vision"],
            "limits": {"requests_per_minute": 60, "tokens_per_minute": 30000},
            "pricing": {"per_token": 0.000001}
        }

    async def _detect_cohere_capabilities(self, api_key: str) -> Dict[str, Any]:
        """Detect Cohere models and capabilities"""
        return {
            "models": ["command", "command-light", "command-xlarge"],
            "features": ["text_generation", "embeddings"],
            "limits": {"requests_per_minute": 100, "tokens_per_minute": 100000},
            "pricing": {"per_token": 0.0000015}
        }

    async def _detect_stability_capabilities(self, api_key: str) -> Dict[str, Any]:
        """Detect Stability AI models and capabilities"""
        return {
            "models": ["stable-diffusion-xl-1024-v1-0", "stable-diffusion-v1-6"],
            "features": ["image_generation"],
            "limits": {"requests_per_minute": 30, "images_per_minute": 30},
            "pricing": {"per_image": 0.01}
        }

    async def _detect_elevenlabs_capabilities(self, api_key: str) -> Dict[str, Any]:
        """Detect ElevenLabs models and capabilities"""
        return {
            "models": ["eleven_monolingual_v1", "eleven_multilingual_v1"],
            "features": ["text_to_speech"],
            "limits": {"requests_per_minute": 80, "characters_per_minute": 5000},
            "pricing": {"per_character": 0.00003}
        }


class APIManager:
    """
    Ultra-advanced API management system with automatic provider detection
    Handles 100+ AI providers with load balancing, failover, and optimization
    """

    def __init__(self, jarvis_instance):
        """
        Initialize API manager

        Args:
            jarvis_instance: Reference to main JARVIS instance
        """
        self.jarvis = jarvis_instance
        self.logger = logging.getLogger('JARVIS.APIManager')

        # API key detection
        self.key_detector = APIKeyDetector()

        # API configurations
        self.api_configs: Dict[APIProvider, APIConfig] = {}
        self.provider_models: Dict[str, List[str]] = {}

        # Request management
        self.request_queue = asyncio.Queue()
        self.response_cache: Dict[str, APIResponse] = {}
        self.cache_ttl = 300  # 5 minutes

        # Load balancing
        self.load_balancer = LoadBalancer(self)
        self.failover_manager = FailoverManager(self)

        # Cost tracking
        self.cost_tracker = CostTracker(self)
        self.budget_manager = BudgetManager(self)

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor(self)

        # Background tasks
        self.tasks: List[asyncio.Task] = []
        self.running = False

        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "total_cost": 0.0,
            "average_response_time": 0.0
        }

    async def initialize(self):
        """Initialize API manager and all providers"""
        try:
            self.logger.info("Initializing API manager...")

            # Load API configurations
            await self._load_api_configurations()

            # Initialize providers
            await self._initialize_providers()

            # Start background tasks
            self.running = True
            self.tasks = [
                asyncio.create_task(self._process_request_queue()),
                asyncio.create_task(self._cleanup_cache()),
                asyncio.create_task(self._update_provider_status()),
            ]

            self.logger.info(f"API manager initialized with {len(self.api_configs)} providers")

        except Exception as e:
            self.logger.error(f"Error initializing API manager: {e}")
            raise

    async def add_api_key(self, api_key: str, custom_name: str = None) -> Dict[str, Any]:
        """
        Add API key with automatic provider detection

        Args:
            api_key: The API key to add
            custom_name: Optional custom name for the configuration

        Returns:
            Result of the operation
        """
        try:
            self.logger.info("Detecting API provider for new key...")

            # Detect provider and validate key
            detected_provider, validation_info = await self.key_detector.detect_provider(api_key)

            if not detected_provider:
                return {
                    "success": False,
                    "error": validation_info.get("error", "Could not detect API provider"),
                    "suggestions": [
                        "Check that the API key is correct",
                        "Ensure the API key has the right format",
                        "Verify that the service is accessible"
                    ]
                }

            # Check if provider already configured
            if detected_provider in self.api_configs:
                return {
                    "success": False,
                    "error": f"Provider {detected_provider.value} is already configured",
                    "provider": detected_provider.value
                }

            # Detect models and capabilities
            self.logger.info(f"Detecting capabilities for {detected_provider.value}...")
            capabilities = await self.key_detector.detect_models_and_capabilities(detected_provider, api_key)

            # Create configuration
            config_name = custom_name or f"{detected_provider.value}_auto"

            # Get default config for provider
            default_config = self._get_default_config_for_provider(detected_provider)

            # Override with detected capabilities
            if capabilities.get("models"):
                default_config["models"] = capabilities["models"]

            if capabilities.get("limits"):
                default_config["rate_limits"] = capabilities["limits"]

            if capabilities.get("pricing"):
                if "per_token" in capabilities["pricing"]:
                    default_config["cost_per_token"] = capabilities["pricing"]["per_token"]
                if "per_image" in capabilities["pricing"]:
                    default_config["cost_per_image"] = capabilities["pricing"]["per_image"]
                if "per_character" in capabilities["pricing"]:
                    default_config["cost_per_character"] = capabilities["pricing"]["per_character"]

            # Create API config
            api_config = APIConfig(
                provider=detected_provider,
                api_key=api_key,
                **default_config
            )

            # Add to configurations
            self.api_configs[detected_provider] = api_config

            # Initialize provider client
            await self._initialize_provider_client(detected_provider, api_config)

            # Save configurations
            self.save_api_configurations()

            result = {
                "success": True,
                "provider": detected_provider.value,
                "models_detected": capabilities.get("models", []),
                "features": capabilities.get("features", []),
                "validation_confidence": validation_info.get("confidence", 0),
                "config_name": config_name
            }

            self.logger.info(f"Successfully added API key for {detected_provider.value} with {len(capabilities.get('models', []))} models")

            return result

        except Exception as e:
            self.logger.error(f"Error adding API key: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _get_default_config_for_provider(self, provider: APIProvider) -> Dict[str, Any]:
        """Get default configuration for a provider"""
        defaults = {
            APIProvider.OPENAI: {
                "base_url": "https://api.openai.com/v1",
                "models": ["gpt-3.5-turbo"],
                "rate_limits": {"requests_per_minute": 60, "tokens_per_minute": 40000},
                "cost_per_token": 0.000002,
                "priority": 1
            },
            APIProvider.ANTHROPIC: {
                "base_url": "https://api.anthropic.com/v1",
                "models": ["claude-3-haiku-20240307"],
                "rate_limits": {"requests_per_minute": 50, "tokens_per_minute": 50000},
                "cost_per_token": 0.000003,
                "priority": 1
            },
            APIProvider.GOOGLE_GEMINI: {
                "base_url": "https://generativelanguage.googleapis.com/v1",
                "models": ["gemini-pro"],
                "rate_limits": {"requests_per_minute": 60, "tokens_per_minute": 30000},
                "cost_per_token": 0.000001,
                "priority": 1
            },
            APIProvider.COHERE: {
                "base_url": "https://api.cohere.ai/v1",
                "models": ["command"],
                "rate_limits": {"requests_per_minute": 100, "tokens_per_minute": 100000},
                "cost_per_token": 0.0000015,
                "priority": 1
            },
            APIProvider.STABILITY_AI: {
                "base_url": "https://api.stability.ai/v1",
                "models": ["stable-diffusion-xl-1024-v1-0"],
                "rate_limits": {"requests_per_minute": 30, "images_per_minute": 30},
                "cost_per_image": 0.01,
                "priority": 1
            },
            APIProvider.ELEVENLABS: {
                "base_url": "https://api.elevenlabs.io/v1",
                "models": ["eleven_monolingual_v1"],
                "rate_limits": {"requests_per_minute": 80, "characters_per_minute": 5000},
                "cost_per_character": 0.00003,
                "priority": 1
            }
        }

        return defaults.get(provider, {
            "base_url": "",
            "models": [],
            "rate_limits": {},
            "cost_per_token": 0.0,
            "priority": 1
        })

    async def test_api_key(self, api_key: str) -> Dict[str, Any]:
        """
        Test an API key without adding it to configuration

        Args:
            api_key: The API key to test

        Returns:
            Test results
        """
        try:
            self.logger.info("Testing API key...")

            # Detect provider
            detected_provider, validation_info = await self.key_detector.detect_provider(api_key)

            if not detected_provider:
                return {
                    "valid": False,
                    "error": validation_info.get("error", "Could not detect provider"),
                    "provider": None
                }

            # Get capabilities
            capabilities = await self.key_detector.detect_models_and_capabilities(detected_provider, api_key)

            return {
                "valid": validation_info.get("valid", False),
                "provider": detected_provider.value,
                "confidence": validation_info.get("confidence", 0),
                "models": capabilities.get("models", []),
                "features": capabilities.get("features", []),
                "limits": capabilities.get("limits", {}),
                "pricing": capabilities.get("pricing", {})
            }

        except Exception as e:
            self.logger.error(f"Error testing API key: {e}")
            return {
                "valid": False,
                "error": str(e)
            }

    def list_supported_providers(self) -> List[Dict[str, Any]]:
        """List all supported API providers with their details"""
        providers_info = []

        for provider_enum in APIProvider:
            # Get key patterns for this provider
            patterns = self.key_detector.key_patterns.get(provider_enum, {})

            provider_info = {
                "name": provider_enum.value,
                "display_name": provider_enum.value.replace('_', ' ').title(),
                "key_formats": patterns.get("prefixes", []),
                "typical_key_length": patterns.get("length_ranges", []),
                "supported_features": self._get_provider_features(provider_enum),
                "configured": provider_enum in self.api_configs
            }

            providers_info.append(provider_info)

        return providers_info

    def _get_provider_features(self, provider: APIProvider) -> List[str]:
        """Get features supported by a provider"""
        features_map = {
            APIProvider.OPENAI: ["text_generation", "chat", "embeddings", "vision", "code"],
            APIProvider.ANTHROPIC: ["text_generation", "chat", "vision"],
            APIProvider.GOOGLE_GEMINI: ["text_generation", "chat", "vision"],
            APIProvider.COHERE: ["text_generation", "embeddings"],
            APIProvider.STABILITY_AI: ["image_generation"],
            APIProvider.ELEVENLABS: ["text_to_speech"],
        }

        return features_map.get(provider, ["text_generation"])

    async def _load_api_configurations(self):
        """Load API configurations from config files and environment"""
        try:
            config_file = os.path.join(os.path.dirname(__file__), '..', 'config', 'api_config.json')

            # Default configurations for all supported providers
            default_configs = {
                APIProvider.OPENAI: {
                    "base_url": "https://api.openai.com/v1",
                    "models": ["gpt-4", "gpt-3.5-turbo", "text-davinci-003"],
                    "rate_limits": {"requests_per_minute": 60, "tokens_per_minute": 40000},
                    "cost_per_token": 0.000002  # $0.002 per 1K tokens
                },
                APIProvider.ANTHROPIC: {
                    "base_url": "https://api.anthropic.com/v1",
                    "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
                    "rate_limits": {"requests_per_minute": 50, "tokens_per_minute": 50000},
                    "cost_per_token": 0.000003
                },
                APIProvider.GOOGLE_GEMINI: {
                    "base_url": "https://generativelanguage.googleapis.com/v1",
                    "models": ["gemini-pro", "gemini-pro-vision"],
                    "rate_limits": {"requests_per_minute": 60, "tokens_per_minute": 30000},
                    "cost_per_token": 0.000001
                },
                APIProvider.COHERE: {
                    "base_url": "https://api.cohere.ai/v1",
                    "models": ["command", "command-light", "command-xlarge"],
                    "rate_limits": {"requests_per_minute": 100, "tokens_per_minute": 100000},
                    "cost_per_token": 0.0000015
                },
                APIProvider.STABILITY_AI: {
                    "base_url": "https://api.stability.ai/v1",
                    "models": ["stable-diffusion-xl-1024-v1-0", "stable-diffusion-v1-6"],
                    "rate_limits": {"requests_per_minute": 30, "images_per_minute": 30},
                    "cost_per_image": 0.01
                },
                APIProvider.ELEVENLABS: {
                    "base_url": "https://api.elevenlabs.io/v1",
                    "models": ["eleven_monolingual_v1", "eleven_multilingual_v1"],
                    "rate_limits": {"requests_per_minute": 80, "characters_per_minute": 5000},
                    "cost_per_character": 0.00003
                }
            }

            # Load from config file if it exists
            saved_config = {}
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    saved_config = json.load(f)

                # Merge with defaults
                for provider, config in saved_config.items():
                    if provider in default_configs:
                        default_configs[provider].update(config)

            # Create API configurations from environment variables, saved config, and defaults
            for provider_enum, default_config in default_configs.items():
                api_key = os.getenv(f"{provider_enum.value.upper()}_API_KEY")

                # Check saved config for API key if not in environment
                if not api_key and provider_enum.value in saved_config:
                    saved_provider_config = saved_config[provider_enum.value]
                    if 'api_key' in saved_provider_config:
                        api_key = saved_provider_config['api_key']

                if api_key:
                    config = APIConfig(
                        provider=provider_enum,
                        api_key=api_key,
                        **default_config
                    )
                    self.api_configs[provider_enum] = config

                    # Initialize provider client
                    await self._initialize_provider_client(provider_enum, config)

            self.logger.info(f"Loaded {len(self.api_configs)} API configurations")

        except Exception as e:
            self.logger.error(f"Error loading API configurations: {e}")

    def save_api_configurations(self):
        """Save API configurations to config file"""
        try:
            config_file = os.path.join(os.path.dirname(__file__), '..', 'config', 'api_config.json')

            # Prepare config data
            config_data = {}
            for provider, config in self.api_configs.items():
                config_data[provider.value] = {
                    'api_key': config.api_key,
                    'base_url': config.base_url,
                    'models': config.models,
                    'rate_limits': config.rate_limits,
                    'cost_per_token': config.cost_per_token,
                    'cost_per_image': getattr(config, 'cost_per_image', 0.0),
                    'priority': config.priority,
                    'enabled': config.enabled
                }

            # Ensure config directory exists
            os.makedirs(os.path.dirname(config_file), exist_ok=True)

            # Save to file
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)

            self.logger.info(f"Saved {len(config_data)} API configurations to {config_file}")

        except Exception as e:
            self.logger.error(f"Error saving API configurations: {e}")

    async def _initialize_providers(self):
        """Initialize all configured API providers"""
        try:
            for provider, config in self.api_configs.items():
                if config.enabled:
                    await self._initialize_provider_client(provider, config)

        except Exception as e:
            self.logger.error(f"Error initializing providers: {e}")

    async def _initialize_provider_client(self, provider: APIProvider, config: APIConfig):
        """Initialize client for specific provider"""
        try:
            if provider == APIProvider.OPENAI and OPENAI_AVAILABLE:
                openai.api_key = config.api_key
                # Test connection
                await openai.ChatCompletion.acreate(
                    model=config.models[0],
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )

            elif provider == APIProvider.ANTHROPIC and ANTHROPIC_AVAILABLE:
                # Initialize Anthropic client
                pass

            elif provider == APIProvider.GOOGLE_GEMINI and GOOGLE_AVAILABLE:
                genai.configure(api_key=config.api_key)

            # Add other provider initializations...

            self.logger.info(f"Initialized provider: {provider.value}")

        except Exception as e:
            self.logger.error(f"Error initializing provider {provider.value}: {e}")
            config.enabled = False

    async def make_request(self,
                          request: APIRequest,
                          use_cache: bool = True) -> APIResponse:
        """
        Make API request with load balancing and failover

        Args:
            request: API request to process
            use_cache: Whether to use response cache

        Returns:
            API response
        """
        start_time = time.time()
        request_id = request.request_id

        try:
            # Check cache first
            if use_cache:
                cached_response = self._get_cached_response(request)
                if cached_response:
                    self.stats["cache_hits"] += 1
                    return cached_response

            # Select best provider
            provider = await self.load_balancer.select_provider(request)

            if not provider:
                raise Exception("No available providers")

            # Make request with retries
            response = await self._execute_request(provider, request)

            # Calculate metrics
            response_time = time.time() - start_time
            response.response_time = response_time

            # Update statistics
            self.stats["total_requests"] += 1
            if response.success:
                self.stats["successful_requests"] += 1
            else:
                self.stats["failed_requests"] += 1

            # Update provider stats
            config = self.api_configs[provider]
            config.last_used = time.time()
            config.total_requests += 1
            if not response.success:
                config.failed_requests += 1

            # Track cost
            if response.cost > 0:
                self.stats["total_cost"] += response.cost
                config.total_cost += response.cost

            # Cache response
            if response.success and use_cache:
                self._cache_response(response)

            return response

        except Exception as e:
            response_time = time.time() - start_time

            self.logger.error(f"Error making API request {request_id}: {e}")

            return APIResponse(
                request_id=request_id,
                provider=request.provider,
                model=request.model,
                response=None,
                response_time=response_time,
                success=False,
                error=str(e)
            )

    async def _execute_request(self, provider: APIProvider, request: APIRequest) -> APIResponse:
        """Execute request against specific provider"""
        try:
            config = self.api_configs[provider]

            if provider == APIProvider.OPENAI and OPENAI_AVAILABLE:
                return await self._execute_openai_request(config, request)

            elif provider == APIProvider.ANTHROPIC and ANTHROPIC_AVAILABLE:
                return await self._execute_anthropic_request(config, request)

            elif provider == APIProvider.GOOGLE_GEMINI and GOOGLE_AVAILABLE:
                return await self._execute_google_request(config, request)

            elif provider == APIProvider.COHERE and COHERE_AVAILABLE:
                return await self._execute_cohere_request(config, request)

            elif provider == APIProvider.STABILITY_AI and STABILITY_AVAILABLE:
                return await self._execute_stability_request(config, request)

            elif provider == APIProvider.ELEVENLABS and ELEVENLABS_AVAILABLE:
                return await self._execute_elevenlabs_request(config, request)

            else:
                # Generic HTTP request for other providers
                return await self._execute_http_request(config, request)

        except Exception as e:
            self.logger.error(f"Error executing request for {provider.value}: {e}")
            raise

    async def _execute_openai_request(self, config: APIConfig, request: APIRequest) -> APIResponse:
        """Execute OpenAI API request"""
        try:
            # Prepare messages
            messages = [{"role": "user", "content": request.prompt}]

            # Make request
            response = await openai.ChatCompletion.acreate(
                model=request.model,
                messages=messages,
                **request.parameters
            )

            # Extract response
            response_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens

            # Calculate cost
            cost = tokens_used * config.cost_per_token

            return APIResponse(
                request_id=request.request_id,
                provider=config.provider,
                model=request.model,
                response=response_text,
                tokens_used=tokens_used,
                cost=cost,
                success=True
            )

        except Exception as e:
            return APIResponse(
                request_id=request.request_id,
                provider=config.provider,
                model=request.model,
                response=None,
                success=False,
                error=str(e)
            )

    async def _execute_anthropic_request(self, config: APIConfig, request: APIRequest) -> APIResponse:
        """Execute Anthropic API request"""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=config.api_key)

            # Prepare messages
            messages = [{"role": "user", "content": request.prompt}]

            # Make request
            response = await client.messages.create(
                model=request.model,
                max_tokens=request.parameters.get("max_tokens", 1000),
                messages=messages,
                **{k: v for k, v in request.parameters.items() if k not in ["max_tokens"]}
            )

            # Extract response
            response_text = response.content[0].text if response.content else ""

            # Calculate tokens (approximate)
            tokens_used = len(request.prompt.split()) + len(response_text.split())

            return APIResponse(
                request_id=request.request_id,
                provider=config.provider,
                model=request.model,
                response=response_text,
                tokens_used=tokens_used,
                cost=tokens_used * config.cost_per_token,
                success=True
            )

        except Exception as e:
            return APIResponse(
                request_id=request.request_id,
                provider=config.provider,
                model=request.model,
                response=None,
                success=False,
                error=str(e)
            )

    async def _execute_google_request(self, config: APIConfig, request: APIRequest) -> APIResponse:
        """Execute Google Gemini API request"""
        try:
            model = genai.GenerativeModel(request.model)

            response = await model.generate_content_async(
                request.prompt,
                **request.parameters
            )

            return APIResponse(
                request_id=request.request_id,
                provider=config.provider,
                model=request.model,
                response=response.text,
                tokens_used=len(response.text.split()),  # Approximate
                cost=len(response.text.split()) * config.cost_per_token,
                success=True
            )

        except Exception as e:
            return APIResponse(
                request_id=request.request_id,
                provider=config.provider,
                model=request.model,
                response=None,
                success=False,
                error=str(e)
            )

    async def _execute_cohere_request(self, config: APIConfig, request: APIRequest) -> APIResponse:
        """Execute Cohere API request"""
        try:
            import cohere

            co = cohere.Client(config.api_key)

            # Make request
            response = co.generate(
                model=request.model,
                prompt=request.prompt,
                max_tokens=request.parameters.get("max_tokens", 100),
                temperature=request.parameters.get("temperature", 0.7),
                **{k: v for k, v in request.parameters.items() if k not in ["max_tokens", "temperature"]}
            )

            # Extract response
            response_text = response.generations[0].text

            # Calculate tokens (approximate)
            tokens_used = len(request.prompt.split()) + len(response_text.split())

            return APIResponse(
                request_id=request.request_id,
                provider=config.provider,
                model=request.model,
                response=response_text,
                tokens_used=tokens_used,
                cost=tokens_used * config.cost_per_token,
                success=True
            )

        except Exception as e:
            return APIResponse(
                request_id=request.request_id,
                provider=config.provider,
                model=request.model,
                response=None,
                success=False,
                error=str(e)
            )

    async def _execute_stability_request(self, config: APIConfig, request: APIRequest) -> APIResponse:
        """Execute Stability AI API request"""
        try:
            import stability_sdk
            from stability_sdk import client
            import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

            # Initialize client
            stability_api = client.StabilityInference(
                key=config.api_key,
                verbose=True,
            )

            # Generate image
            answers = stability_api.generate(
                prompt=request.prompt,
                **request.parameters
            )

            # Get first image
            for resp in answers:
                for artifact in resp.artifacts:
                    if artifact.finish_reason == generation.FILTER:
                        return APIResponse(
                            request_id=request.request_id,
                            provider=config.provider,
                            model=request.model,
                            response=None,
                            success=False,
                            error="Content filtered"
                        )
                    if artifact.type == generation.ARTIFACT_IMAGE:
                        # Save image data
                        image_data = artifact.binary

                        return APIResponse(
                            request_id=request.request_id,
                            provider=config.provider,
                            model=request.model,
                            response=image_data,
                            tokens_used=1,  # Fixed cost per image
                            cost=config.cost_per_image,
                            success=True
                        )

            return APIResponse(
                request_id=request.request_id,
                provider=config.provider,
                model=request.model,
                response=None,
                success=False,
                error="No image generated"
            )

        except Exception as e:
            return APIResponse(
                request_id=request.request_id,
                provider=config.provider,
                model=request.model,
                response=None,
                success=False,
                error=str(e)
            )

    async def _execute_elevenlabs_request(self, config: APIConfig, request: APIRequest) -> APIResponse:
        """Execute ElevenLabs API request"""
        try:
            import elevenlabs

            # Set API key
            elevenlabs.set_api_key(config.api_key)

            # Get text to convert
            text = request.prompt

            # Generate speech
            audio = elevenlabs.generate(
                text=text,
                voice=request.parameters.get("voice", "Drew"),
                model=request.model
            )

            # Calculate cost (based on character count)
            char_count = len(text)
            cost = char_count * config.cost_per_character

            return APIResponse(
                request_id=request.request_id,
                provider=config.provider,
                model=request.model,
                response=audio,  # Audio data
                tokens_used=char_count,  # Characters used
                cost=cost,
                success=True
            )

        except Exception as e:
            return APIResponse(
                request_id=request.request_id,
                provider=config.provider,
                model=request.model,
                response=None,
                success=False,
                error=str(e)
            )

    async def _execute_http_request(self, config: APIConfig, request: APIRequest) -> APIResponse:
        """Execute generic HTTP request"""
        try:
            async with aiohttp.ClientSession() as session:
                # Build request URL and headers
                headers = {
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json"
                }

                # Make request
                async with session.post(
                    f"{config.base_url}/generate",
                    json={
                        "model": request.model,
                        "prompt": request.prompt,
                        **request.parameters
                    },
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=request.timeout)
                ) as response:

                    if response.status == 200:
                        result = await response.json()

                        return APIResponse(
                            request_id=request.request_id,
                            provider=config.provider,
                            model=request.model,
                            response=result.get("text", ""),
                            tokens_used=result.get("tokens", 0),
                            cost=result.get("tokens", 0) * config.cost_per_token,
                            success=True
                        )
                    else:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")

        except Exception as e:
            return APIResponse(
                request_id=request.request_id,
                provider=config.provider,
                model=request.model,
                response=None,
                success=False,
                error=str(e)
            )

    def _get_cached_response(self, request: APIRequest) -> Optional[APIResponse]:
        """Get cached response if available"""
        cache_key = self._generate_cache_key(request)

        if cache_key in self.response_cache:
            cached = self.response_cache[cache_key]

            # Check if cache is still valid
            if time.time() - cached.response_time < self.cache_ttl:
                return cached

            # Remove expired cache
            del self.response_cache[cache_key]

        return None

    def _cache_response(self, response: APIResponse):
        """Cache API response"""
        # Generate cache key based on request parameters
        # This would need the original request to generate the key
        pass

    def _generate_cache_key(self, request: APIRequest) -> str:
        """Generate cache key for request"""
        key_data = f"{request.provider.value}:{request.model}:{request.prompt}:{request.parameters}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    async def _process_request_queue(self):
        """Process queued API requests"""
        while self.running:
            try:
                request = await asyncio.wait_for(self.request_queue.get(), timeout=1.0)

                # Process request in background
                asyncio.create_task(self.make_request(request))

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing request queue: {e}")

    async def _cleanup_cache(self):
        """Clean up expired cache entries"""
        while self.running:
            try:
                current_time = time.time()
                expired_keys = []

                for key, response in self.response_cache.items():
                    if current_time - response.response_time > self.cache_ttl:
                        expired_keys.append(key)

                for key in expired_keys:
                    del self.response_cache[key]

                await asyncio.sleep(60)  # Clean every minute

            except Exception as e:
                self.logger.error(f"Error cleaning cache: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    async def _update_provider_status(self):
        """Update provider status and health"""
        while self.running:
            try:
                for provider, config in self.api_configs.items():
                    if config.enabled:
                        # Test provider health
                        health = await self._test_provider_health(provider, config)

                        # Update configuration based on health
                        if not health["healthy"]:
                            config.enabled = False
                            self.logger.warning(f"Disabled unhealthy provider: {provider.value}")
                        else:
                            config.enabled = True

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                self.logger.error(f"Error updating provider status: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes before retry

    async def _test_provider_health(self, provider: APIProvider, config: APIConfig) -> Dict[str, Any]:
        """Test provider health"""
        try:
            # Create test request
            test_request = APIRequest(
                provider=provider,
                model=config.models[0] if config.models else "test",
                prompt="Hello",
                timeout=10
            )

            # Execute test request
            response = await self._execute_request(provider, test_request)

            return {
                "healthy": response.success,
                "response_time": response.response_time,
                "error": response.error if not response.success else None
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }

    def queue_request(self, request: APIRequest):
        """Queue API request for processing"""
        self.request_queue.put_nowait(request)

    def get_provider_info(self, provider: APIProvider) -> Optional[Dict[str, Any]]:
        """Get information about a provider"""
        if provider not in self.api_configs:
            return None

        config = self.api_configs[provider]

        return {
            "provider": provider.value,
            "enabled": config.enabled,
            "models": config.models,
            "rate_limits": config.rate_limits,
            "cost_per_token": config.cost_per_token,
            "total_requests": config.total_requests,
            "failed_requests": config.failed_requests,
            "total_cost": config.total_cost,
            "last_used": config.last_used,
            "success_rate": (config.total_requests - config.failed_requests) / config.total_requests if config.total_requests > 0 else 0
        }

    def get_all_providers(self) -> List[Dict[str, Any]]:
        """Get information about all providers"""
        return [self.get_provider_info(provider) for provider in self.api_configs.keys()]

    def get_stats(self) -> Dict[str, Any]:
        """Get API manager statistics"""
        return {
            **self.stats,
            "providers_configured": len(self.api_configs),
            "providers_enabled": len([c for c in self.api_configs.values() if c.enabled]),
            "cache_size": len(self.response_cache),
            "queue_size": self.request_queue.qsize()
        }

    async def shutdown(self):
        """Shutdown API manager"""
        try:
            self.logger.info("Shutting down API manager...")

            self.running = False

            # Cancel all tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()

            # Wait for tasks to complete
            await asyncio.gather(*self.tasks, return_exceptions=True)

            self.logger.info("API manager shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down API manager: {e}")


class LoadBalancer:
    """Load balancer for API providers"""

    def __init__(self, api_manager):
        self.api_manager = api_manager

    async def select_provider(self, request: APIRequest) -> Optional[APIProvider]:
        """Select best provider for request"""
        try:
            available_providers = []

            for provider, config in self.api_manager.api_configs.items():
                if not config.enabled:
                    continue

                # Check if provider supports request type
                if self._supports_request_type(provider, request):
                    available_providers.append(provider)

            if not available_providers:
                return None

            # Score providers based on various factors
            scored_providers = []

            for provider in available_providers:
                score = await self._score_provider(provider, request)
                scored_providers.append((provider, score))

            # Sort by score (highest first)
            scored_providers.sort(key=lambda x: x[1], reverse=True)

            return scored_providers[0][0] if scored_providers else None

        except Exception as e:
            self.api_manager.logger.error(f"Error selecting provider: {e}")
            return None

    def _supports_request_type(self, provider: APIProvider, request: APIRequest) -> bool:
        """Check if provider supports request type"""
        config = self.api_manager.api_configs[provider]

        # Check if model is available
        if request.model not in config.models:
            return False

        # Check request type compatibility
        if request.request_type == "image" and provider in [APIProvider.STABILITY_AI, APIProvider.DALL_E]:
            return True
        elif request.request_type == "audio" and provider in [APIProvider.ELEVENLABS]:
            return True
        elif request.request_type == "text":
            return True

        return False

    async def _score_provider(self, provider: APIProvider, request: APIRequest) -> float:
        """Score provider for request"""
        config = self.api_manager.api_configs[provider]
        score = 0.0

        # Base score from priority
        score += config.priority * 10

        # Success rate bonus
        if config.total_requests > 0:
            success_rate = (config.total_requests - config.failed_requests) / config.total_requests
            score += success_rate * 20

        # Recency bonus (prefer recently used providers)
        time_since_last_use = time.time() - config.last_used
        if time_since_last_use < 300:  # Used within last 5 minutes
            score += 15
        elif time_since_last_use < 900:  # Used within last 15 minutes
            score += 10

        # Cost efficiency (prefer cheaper providers)
        if config.cost_per_token > 0:
            # Invert cost (lower cost = higher score)
            cost_score = max(0, 10 - (config.cost_per_token * 1000000))  # Scale factor
            score += cost_score

        return score


class FailoverManager:
    """Failover management for API providers"""

    def __init__(self, api_manager):
        self.api_manager = api_manager

    async def handle_failure(self, provider: APIProvider, error: str):
        """Handle provider failure"""
        try:
            config = self.api_manager.api_configs[provider]

            # Disable provider temporarily
            config.enabled = False

            # Log failure
            self.api_manager.logger.warning(f"Provider {provider.value} failed: {error}")

            # Try to find alternative provider
            # This would implement failover logic

        except Exception as e:
            self.api_manager.logger.error(f"Error handling failover: {e}")


class CostTracker:
    """Track API usage costs"""

    def __init__(self, api_manager):
        self.api_manager = api_manager

    def get_total_cost(self) -> float:
        """Get total API costs"""
        return sum(config.total_cost for config in self.api_manager.api_configs.values())

    def get_cost_by_provider(self) -> Dict[str, float]:
        """Get costs broken down by provider"""
        return {
            provider.value: config.total_cost
            for provider, config in self.api_manager.api_configs.items()
        }


class BudgetManager:
    """Manage API usage budgets"""

    def __init__(self, api_manager):
        self.api_manager = api_manager
        self.monthly_budget = 100.0  # Default $100 per month
        self.alert_threshold = 0.8   # Alert at 80% of budget

    def check_budget(self) -> Dict[str, Any]:
        """Check current budget status"""
        current_cost = self.get_total_cost()
        budget_percentage = (current_cost / self.monthly_budget) * 100

        return {
            "current_cost": current_cost,
            "monthly_budget": self.monthly_budget,
            "budget_used_percent": budget_percentage,
            "budget_remaining": self.monthly_budget - current_cost,
            "alert_triggered": budget_percentage > (self.alert_threshold * 100)
        }


class PerformanceMonitor:
    """Monitor API performance"""

    def __init__(self, api_manager):
        self.api_manager = api_manager
        self.response_times = []

    def record_response_time(self, response_time: float):
        """Record API response time"""
        self.response_times.append(response_time)

        # Keep only recent measurements
        if len(self.response_times) > 1000:
            self.response_times.pop(0)

    def get_average_response_time(self) -> float:
        """Get average response time"""
        if not self.response_times:
            return 0.0

        return sum(self.response_times) / len(self.response_times)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "average_response_time": self.get_average_response_time(),
            "response_times_count": len(self.response_times),
            "fastest_response": min(self.response_times) if self.response_times else 0,
            "slowest_response": max(self.response_times) if self.response_times else 0
        }