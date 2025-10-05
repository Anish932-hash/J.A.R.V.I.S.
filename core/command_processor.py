"""
J.A.R.V.I.S. Command Processor
Advanced command processing and natural language understanding
"""

import re
import time
import json
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
import difflib


class Command:
    """Command class for processing user commands"""

    def __init__(self,
                 text: str,
                 confidence: float = 1.0,
                 context: Dict[str, Any] = None,
                 source: str = "text",
                 command_id: str = None):
        """
        Initialize command

        Args:
            text: Command text
            confidence: Confidence score (0.0 to 1.0)
            context: Additional context
            source: Command source (text, voice, etc.)
            command_id: Unique command identifier
        """
        self.command_id = command_id or f"cmd_{time.time()}_{hash(text)}"
        self.text = text.lower().strip()
        self.original_text = text
        self.confidence = confidence
        self.context = context or {}
        self.source = source
        self.timestamp = time.time()
        self.created_at = datetime.now().isoformat()

        # Processing results
        self.parsed_intent = None
        self.entities = []
        self.response = None
        self.execution_result = None
        self.processing_time = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert command to dictionary"""
        return {
            "command_id": self.command_id,
            "text": self.text,
            "original_text": self.original_text,
            "confidence": self.confidence,
            "context": self.context,
            "source": self.source,
            "timestamp": self.timestamp,
            "created_at": self.created_at,
            "parsed_intent": self.parsed_intent,
            "entities": self.entities,
            "response": self.response,
            "execution_result": self.execution_result,
            "processing_time": self.processing_time
        }


class CommandProcessor:
    """
    Advanced command processing system
    Handles natural language understanding and command execution
    """

    def __init__(self, jarvis_instance):
        """
        Initialize command processor

        Args:
            jarvis_instance: Reference to main JARVIS instance
        """
        self.jarvis = jarvis_instance

        # Command patterns and handlers
        self.command_patterns = {}
        self.intent_handlers = {}
        self.entity_extractors = {}

        # Command history and learning
        self.command_history = []
        self.max_history_size = 1000
        self.command_suggestions = {}

        # Processing queues
        self.pending_commands = []
        self.processing_commands = set()

        # Performance tracking
        self.processing_stats = {
            "total_commands": 0,
            "successful_commands": 0,
            "failed_commands": 0,
            "average_processing_time": 0.0
        }

        # Initialize built-in commands
        self._initialize_builtin_commands()

    def _initialize_builtin_commands(self):
        """Initialize built-in command patterns"""
        # System commands
        self.register_command_pattern(
            r"(?:jarvis|hey jarvis|okay jarvis)",
            "system.wake",
            ["wake_word"]
        )

        self.register_command_pattern(
            r"(?:shutdown|shut down|turn off|exit|quit|bye|goodbye)",
            "system.shutdown",
            ["shutdown_command"]
        )

        self.register_command_pattern(
            r"(?:restart|reboot)",
            "system.restart",
            ["restart_command"]
        )

        self.register_command_pattern(
            r"(?:status|how are you|what's up)",
            "system.status",
            ["status_query"]
        )

        # Application commands
        self.register_command_pattern(
            r"(?:open|launch|start) (.+)",
            "application.launch",
            ["application_name"]
        )

        self.register_command_pattern(
            r"(?:close|quit|exit) (.+)",
            "application.close",
            ["application_name"]
        )

        # File commands
        self.register_command_pattern(
            r"(?:create|make|new) (?:file|document) (.+)",
            "file.create",
            ["file_path"]
        )

        self.register_command_pattern(
            r"(?:delete|remove) (?:file|document) (.+)",
            "file.delete",
            ["file_path"]
        )

        self.register_command_pattern(
            r"(?:search|find) (?:for )?(.+)",
            "file.search",
            ["search_query"]
        )

        # System info commands
        self.register_command_pattern(
            r"(?:what is|tell me) (?:the )?(?:cpu|memory|disk|system) (?:usage|status)",
            "system.info",
            ["info_type"]
        )

        # Volume commands
        self.register_command_pattern(
            r"(?:set|change) volume (?:to )?(\d+)",
            "system.volume",
            ["volume_level"]
        )

        # Time and date commands
        self.register_command_pattern(
            r"(?:what time|current time|time is it)",
            "system.time",
            ["time_query"]
        )

        self.register_command_pattern(
            r"(?:what day|what date|today)",
            "system.date",
            ["date_query"]
        )

        # API Management commands
        self.register_command_pattern(
            r"(?:add|configure|setup) (?:api|API) key (.+)",
            "api.add_key",
            ["api_key"]
        )

        self.register_command_pattern(
            r"(?:test|validate|check) (?:api|API) key (.+)",
            "api.test_key",
            ["api_key"]
        )

        self.register_command_pattern(
            r"(?:list|show) (?:api|API) providers",
            "api.list_providers",
            []
        )

        self.register_command_pattern(
            r"(?:remove|delete) (?:api|API) (?:provider|key) (.+)",
            "api.remove_provider",
            ["provider_name"]
        )

        # Self-development commands
        # self.register_command_pattern(
        #     r"(?:develop|create|build) (?:a |an |)(?:feature|functionality) (.+)",
        #     "development.create_feature",
        #     ["feature_description"]
        # )

        # self.register_command_pattern(
        #     r"(?:fix|resolve|correct) (?:bug|issue|problem) (.+)",
        #     "development.fix_bug",
        #     ["bug_description"]
        # )

        # self.register_command_pattern(
        #     r"(?:optimize|improve|enhance) (.+)",
        #     "development.optimize",
        #     ["optimization_target"]
        # )

        # self.register_command_pattern(
        #     r"(?:research|learn about|study) (.+)",
        #     "development.research",
        #     ["research_topic"]
        # )

        # self.register_command_pattern(
        #     r"(?:search|find) (?:information|data|web) (?:about|on|for) (.+)",
        #     "development.web_search",
        #     ["search_query"]
        # )

        # self.register_command_pattern(
        #     r"(?:test|run tests|check) (.+)",
        #     "development.test",
        #     ["test_target"]
        # )

        # self.register_command_pattern(
        #     r"(?:validate|check|verify) (?:code|implementation) (.+)",
        #     "development.validate",
        #     ["validation_target"]
        # )

        # self.register_command_pattern(
        #     r"(?:think|reason|analyze) (?:about|on) (.+)",
        #     "development.reason",
        #     ["reasoning_topic"]
        # )

    def register_command_pattern(self,
                                pattern: str,
                                intent: str,
                                entities: List[str],
                                handler: Callable = None):
        """
        Register a command pattern

        Args:
            pattern: Regex pattern to match
            intent: Intent identifier
            entities: List of entity names to extract
            handler: Optional handler function
        """
        compiled_pattern = re.compile(pattern, re.IGNORECASE)

        self.command_patterns[intent] = {
            "pattern": compiled_pattern,
            "entities": entities,
            "handler": handler
        }

        if handler:
            self.intent_handlers[intent] = handler

    def register_intent_handler(self, intent: str, handler: Callable):
        """
        Register an intent handler

        Args:
            intent: Intent identifier
            handler: Handler function
        """
        self.intent_handlers[intent] = handler

    def register_command(self, command_name: str, handler: Callable):
        """
        Register a command handler (convenience method)

        Args:
            command_name: Command name
            handler: Handler function
        """
        # Register as a simple pattern that matches the command name
        pattern = rf"\b{re.escape(command_name)}\b"
        self.register_command_pattern(pattern, command_name, [], handler)

    def process_command(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a command

        Args:
            text: Command text
            context: Additional context

        Returns:
            Processing result
        """
        start_time = time.time()

        try:
            # Create command object
            command = Command(text, context=context)

            # Add to history
            self._add_to_history(command)

            # Parse command
            parsed = self._parse_command(command)

            # Execute command
            result = self._execute_command(parsed)

            # Update statistics
            processing_time = time.time() - start_time
            self._update_processing_stats(True, processing_time)

            return {
                "success": True,
                "command": command.to_dict(),
                "result": result,
                "processing_time": processing_time
            }

        except Exception as e:
            processing_time = time.time() - start_time
            self._update_processing_stats(False, processing_time)

            self.jarvis.logger.error(f"Error processing command '{text}': {e}")

            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time
            }

    def process_pending_commands(self):
        """Process pending commands (called from main loop)"""
        if not self.pending_commands:
            return

        # Process up to 5 commands per cycle
        for _ in range(min(5, len(self.pending_commands))):
            try:
                command_text, context, callback = self.pending_commands.pop(0)

                if command_text not in self.processing_commands:
                    self.processing_commands.add(command_text)

                    # Process in background thread
                    thread = threading.Thread(
                        target=self._process_async_command,
                        args=(command_text, context, callback),
                        daemon=True
                    )
                    thread.start()

            except Exception as e:
                self.jarvis.logger.error(f"Error processing pending command: {e}")

    def _process_async_command(self, text: str, context: Dict[str, Any], callback: Callable):
        """Process command asynchronously"""
        try:
            result = self.process_command(text, context)

            if callback:
                callback(result)

        except Exception as e:
            self.jarvis.logger.error(f"Error in async command processing: {e}")
        finally:
            self.processing_commands.discard(text)

    def execute_command(self, command: str, context: Dict[str, Any] = None) -> Any:
        """
        Execute a command synchronously

        Args:
            command: Command to execute
            context: Additional context

        Returns:
            Execution result
        """
        result = self.process_command(command, context)

        if result["success"]:
            return result["result"]
        else:
            raise Exception(result.get("error", "Command execution failed"))

    def add_pending_command(self, command: str, context: Dict[str, Any] = None, callback: Callable = None):
        """
        Add command to pending queue

        Args:
            command: Command text
            context: Additional context
            callback: Optional callback for result
        """
        self.pending_commands.append((command, context, callback))

    def _parse_command(self, command: Command) -> Command:
        """Parse command to extract intent and entities"""
        best_match = None
        best_confidence = 0.0

        # Try to match against patterns
        for intent, pattern_info in self.command_patterns.items():
            pattern = pattern_info["pattern"]
            match = pattern.search(command.text)

            if match:
                confidence = self._calculate_confidence(command.text, intent)

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = {
                        "intent": intent,
                        "entities": self._extract_entities(match, pattern_info["entities"]),
                        "confidence": confidence,
                        "match": match
                    }

        if best_match:
            command.parsed_intent = best_match["intent"]
            command.entities = best_match["entities"]
            command.confidence = best_match["confidence"]

            # Update suggestions for similar commands
            self._update_suggestions(command.text, best_match["intent"])

        return command

    def _calculate_confidence(self, text: str, intent: str) -> float:
        """Calculate confidence score for command matching using advanced NLP"""
        try:
            # Import transformer models for semantic similarity
            from sentence_transformers import SentenceTransformer, util

            # Initialize model (lazy loading for performance)
            if not hasattr(self, '_semantic_model'):
                self._semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Check for exact matches first
            if text in self.command_suggestions:
                return 0.95

            # Use transformer-based semantic similarity for better understanding
            if hasattr(self, '_semantic_model'):
                # Get intent description/pattern
                intent_pattern = self.command_patterns.get(intent, {}).get("pattern", "")
                if intent_pattern:
                    # Get embeddings
                    text_embedding = self._semantic_model.encode(text, convert_to_tensor=True)
                    intent_embedding = self._semantic_model.encode(str(intent_pattern.pattern), convert_to_tensor=True)

                    # Calculate cosine similarity
                    similarity = util.pytorch_cos_sim(text_embedding, intent_embedding).item()

                    # Return scaled similarity as confidence
                    return min(0.95, max(0.5, similarity))

            # Fallback to fuzzy matching
            similar_commands = difflib.get_close_matches(
                text,
                [cmd["text"] for cmd in self.command_history[-100:]],
                n=1,
                cutoff=0.6
            )

            if similar_commands:
                return 0.7

            # Default confidence for pattern matches
            return 0.6

        except ImportError:
            # Fallback if sentence_transformers not available
            if text in self.command_suggestions:
                return 0.9

            similar_commands = difflib.get_close_matches(
                text,
                [cmd["text"] for cmd in self.command_history[-100:]],
                n=1,
                cutoff=0.6
            )

            if similar_commands:
                return 0.7

            return 0.6
        except Exception as e:
            self.jarvis.logger.warning(f"Error in confidence calculation: {e}")
            return 0.6

    def _extract_entities(self, match: re.Match, entity_names: List[str]) -> List[Dict[str, Any]]:
        """Extract entities from regex match with advanced NER"""
        entities = []

        # Basic regex extraction
        for i, entity_name in enumerate(entity_names):
            try:
                if i + 1 <= len(match.groups()):
                    entity_value = match.group(i + 1)
                    entities.append({
                        "name": entity_name,
                        "value": entity_value,
                        "start": match.start(i + 1),
                        "end": match.end(i + 1),
                        "confidence": 0.9
                    })
            except:
                pass

        # Enhanced entity extraction using spaCy for NER
        try:
            import spacy
            if not hasattr(self, '_nlp_model'):
                try:
                    self._nlp_model = spacy.load("en_core_web_sm")
                except OSError:
                    # Model not installed, skip advanced NER
                    return entities

            # Process full text with spaCy
            doc = self._nlp_model(match.string)

            # Extract named entities
            for ent in doc.ents:
                # Map spaCy entity types to our entity types
                entity_type = self._map_spacy_entity_type(ent.label_)
                if entity_type:
                    entities.append({
                        "name": entity_type,
                        "value": ent.text,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": 0.85,
                        "spacy_label": ent.label_
                    })

        except ImportError:
            # spaCy not available, use basic extraction
            pass
        except Exception as e:
            self.jarvis.logger.debug(f"Error in advanced entity extraction: {e}")

        return entities

    def _map_spacy_entity_type(self, spacy_label: str) -> Optional[str]:
        """Map spaCy entity labels to command entity types"""
        mapping = {
            "PERSON": "person_name",
            "ORG": "organization",
            "GPE": "location",
            "DATE": "date",
            "TIME": "time",
            "MONEY": "amount",
            "PERCENT": "percentage",
            "CARDINAL": "number",
            "ORDINAL": "ordinal",
            "PRODUCT": "product_name",
            "EVENT": "event_name",
            "WORK_OF_ART": "title",
            "FAC": "facility",
            "LOC": "location"
        }
        return mapping.get(spacy_label)

    def _execute_command(self, command: Command) -> Any:
        """Execute parsed command"""
        if not command.parsed_intent:
            return self._handle_unknown_command(command)

        # Get handler for intent
        handler = self.intent_handlers.get(command.parsed_intent)

        if handler:
            try:
                return handler(command)
            except Exception as e:
                self.jarvis.logger.error(f"Error in command handler: {e}")
                return self._handle_command_error(command, str(e))

        # Default execution based on intent
        return self._execute_builtin_command(command)

    def _execute_builtin_command(self, command: Command) -> Any:
        """Execute built-in commands"""
        intent = command.parsed_intent

        # System commands
        if intent == "system.status":
            return self._handle_system_status(command)
        elif intent == "system.shutdown":
            return self._handle_system_shutdown(command)
        elif intent == "system.restart":
            return self._handle_system_restart(command)
        elif intent == "system.time":
            return self._handle_system_time(command)
        elif intent == "system.date":
            return self._handle_system_date(command)
        elif intent == "system.volume":
            return self._handle_system_volume(command)

        # Application commands
        elif intent == "application.launch":
            return self._handle_application_launch(command)
        elif intent == "application.close":
            return self._handle_application_close(command)

        # File commands
        elif intent == "file.create":
            return self._handle_file_create(command)
        elif intent == "file.delete":
            return self._handle_file_delete(command)
        elif intent == "file.search":
            return self._handle_file_search(command)

        # System info commands
        elif intent == "system.info":
            return self._handle_system_info(command)

        # API Management commands
        elif intent == "api.add_key":
            return self._handle_api_add_key(command)
        elif intent == "api.test_key":
            return self._handle_api_test_key(command)
        elif intent == "api.list_providers":
            return self._handle_api_list_providers(command)
        elif intent == "api.remove_provider":
            return self._handle_api_remove_provider(command)

        # Self-development commands
        # elif intent == "development.create_feature":
        #     return self._handle_development_create_feature(command)
        # elif intent == "development.fix_bug":
        #     return self._handle_development_fix_bug(command)
        # elif intent == "development.optimize":
        #     return self._handle_development_optimize(command)
        # elif intent == "development.research":
        #     return self._handle_development_research(command)
        # elif intent == "development.web_search":
        #     return self._handle_development_web_search(command)
        # elif intent == "development.test":
        #     return self._handle_development_test(command)
        # elif intent == "development.validate":
        #     return self._handle_development_validate(command)
        # elif intent == "development.reason":
        #     return self._handle_development_reason(command)

        else:
            return self._handle_unknown_intent(command)

    def _handle_system_status(self, command: Command) -> Dict[str, Any]:
        """Handle system status command"""
        return {
            "action": "system_status",
            "message": "System is operational",
            "data": self.jarvis.get_status()
        }

    def _handle_system_shutdown(self, command: Command) -> Dict[str, Any]:
        """Handle system shutdown command"""
        self.jarvis.shutdown()
        return {
            "action": "system_shutdown",
            "message": "Shutting down J.A.R.V.I.S."
        }

    def _handle_system_restart(self, command: Command) -> Dict[str, Any]:
        """Handle system restart command"""
        # This would restart the JARVIS process
        return {
            "action": "system_restart",
            "message": "Restarting J.A.R.V.I.S.",
            "restart_required": True
        }

    def _handle_system_time(self, command: Command) -> Dict[str, Any]:
        """Handle time query command"""
        current_time = datetime.now().strftime("%I:%M %p")
        return {
            "action": "system_time",
            "message": f"Current time is {current_time}",
            "data": {"time": current_time}
        }

    def _handle_system_date(self, command: Command) -> Dict[str, Any]:
        """Handle date query command"""
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        return {
            "action": "system_date",
            "message": f"Today is {current_date}",
            "data": {"date": current_date}
        }

    def _handle_system_volume(self, command: Command) -> Dict[str, Any]:
        """Handle volume control command"""
        # Extract volume level from entities
        volume_level = 50  # default
        for entity in command.entities:
            if entity["name"] == "volume_level":
                try:
                    volume_level = int(entity["value"])
                    volume_level = max(0, min(100, volume_level))  # Clamp to 0-100
                except:
                    pass

        # This would use Windows audio APIs to set volume
        return {
            "action": "system_volume",
            "message": f"Setting volume to {volume_level}%",
            "data": {"volume_level": volume_level}
        }

    def _handle_application_launch(self, command: Command) -> Dict[str, Any]:
        """Handle application launch command"""
        app_name = "unknown"
        for entity in command.entities:
            if entity["name"] == "application_name":
                app_name = entity["value"]
                break

        # This would use the application controller
        return {
            "action": "application_launch",
            "message": f"Launching {app_name}",
            "data": {"application": app_name}
        }

    def _handle_application_close(self, command: Command) -> Dict[str, Any]:
        """Handle application close command"""
        app_name = "unknown"
        for entity in command.entities:
            if entity["name"] == "application_name":
                app_name = entity["value"]
                break

        return {
            "action": "application_close",
            "message": f"Closing {app_name}",
            "data": {"application": app_name}
        }

    def _handle_file_create(self, command: Command) -> Dict[str, Any]:
        """Handle file creation command"""
        file_path = ""
        for entity in command.entities:
            if entity["name"] == "file_path":
                file_path = entity["value"]
                break

        return {
            "action": "file_create",
            "message": f"Creating file: {file_path}",
            "data": {"file_path": file_path}
        }

    def _handle_file_delete(self, command: Command) -> Dict[str, Any]:
        """Handle file deletion command"""
        file_path = ""
        for entity in command.entities:
            if entity["name"] == "file_path":
                file_path = entity["value"]
                break

        return {
            "action": "file_delete",
            "message": f"Deleting file: {file_path}",
            "data": {"file_path": file_path}
        }

    def _handle_file_search(self, command: Command) -> Dict[str, Any]:
        """Handle file search command"""
        query = ""
        for entity in command.entities:
            if entity["name"] == "search_query":
                query = entity["value"]
                break

        return {
            "action": "file_search",
            "message": f"Searching for: {query}",
            "data": {"query": query}
        }

    def _handle_system_info(self, command: Command) -> Dict[str, Any]:
        """Handle system information command"""
        info_type = "general"
        for entity in command.entities:
            if entity["name"] == "info_type":
                info_type = entity["value"]
                break

        return {
            "action": "system_info",
            "message": f"Getting {info_type} information",
            "data": {"info_type": info_type}
        }

    def _handle_development_create_feature(self, command: Command) -> Dict[str, Any]:
        """Handle feature creation command"""
        feature_description = ""
        for entity in command.entities:
            if entity["name"] == "feature_description":
                feature_description = entity["value"]
                break

        if self.jarvis.self_development_engine:
            try:
                task_id = self.jarvis.self_development_engine.create_task(
                    task_type="feature",
                    description=feature_description,
                    priority=5
                )
                return {
                    "action": "development_create_feature",
                    "message": f"Started developing feature: {feature_description}",
                    "data": {"task_id": task_id, "feature": feature_description}
                }
            except Exception as e:
                return {
                    "action": "development_error",
                    "message": f"Error creating feature: {str(e)}",
                    "data": {"error": str(e)}
                }
        else:
            return {
                "action": "development_unavailable",
                "message": "Self-development engine is not available",
                "data": {}
            }

    def _handle_development_fix_bug(self, command: Command) -> Dict[str, Any]:
        """Handle bug fix command"""
        bug_description = ""
        for entity in command.entities:
            if entity["name"] == "bug_description":
                bug_description = entity["value"]
                break

        if self.jarvis.self_development_engine:
            try:
                task_id = self.jarvis.self_development_engine.create_task(
                    task_type="bug_fix",
                    description=bug_description,
                    priority=7  # Higher priority for bug fixes
                )
                return {
                    "action": "development_fix_bug",
                    "message": f"Started fixing bug: {bug_description}",
                    "data": {"task_id": task_id, "bug": bug_description}
                }
            except Exception as e:
                return {
                    "action": "development_error",
                    "message": f"Error fixing bug: {str(e)}",
                    "data": {"error": str(e)}
                }
        else:
            return {
                "action": "development_unavailable",
                "message": "Self-development engine is not available",
                "data": {}
            }

    def _handle_development_optimize(self, command: Command) -> Dict[str, Any]:
        """Handle optimization command"""
        optimization_target = ""
        for entity in command.entities:
            if entity["name"] == "optimization_target":
                optimization_target = entity["value"]
                break

        if self.jarvis.self_development_engine:
            try:
                task_id = self.jarvis.self_development_engine.create_task(
                    task_type="optimization",
                    description=f"Optimize {optimization_target}",
                    priority=4
                )
                return {
                    "action": "development_optimize",
                    "message": f"Started optimizing: {optimization_target}",
                    "data": {"task_id": task_id, "target": optimization_target}
                }
            except Exception as e:
                return {
                    "action": "development_error",
                    "message": f"Error optimizing: {str(e)}",
                    "data": {"error": str(e)}
                }
        else:
            return {
                "action": "development_unavailable",
                "message": "Self-development engine is not available",
                "data": {}
            }

    def _handle_development_research(self, command: Command) -> Dict[str, Any]:
        """Handle research command"""
        research_topic = ""
        for entity in command.entities:
            if entity["name"] == "research_topic":
                research_topic = entity["value"]
                break

        if self.jarvis.self_development_engine:
            try:
                task_id = self.jarvis.self_development_engine.create_task(
                    task_type="research",
                    description=f"Research {research_topic}",
                    priority=3
                )
                return {
                    "action": "development_research",
                    "message": f"Started researching: {research_topic}",
                    "data": {"task_id": task_id, "topic": research_topic}
                }
            except Exception as e:
                return {
                    "action": "development_error",
                    "message": f"Error researching: {str(e)}",
                    "data": {"error": str(e)}
                }
        else:
            return {
                "action": "development_unavailable",
                "message": "Self-development engine is not available",
                "data": {}
            }

    def _handle_development_web_search(self, command: Command) -> Dict[str, Any]:
        """Handle web search command"""
        search_query = ""
        for entity in command.entities:
            if entity["name"] == "search_query":
                search_query = entity["value"]
                break

        if self.jarvis.self_development_engine and hasattr(self.jarvis.self_development_engine, 'web_searcher'):
            try:
                # Perform immediate search
                search_results = self.jarvis.self_development_engine.web_searcher.search(
                    query=search_query,
                    max_results=5,
                    include_content=True
                )

                return {
                    "action": "development_web_search",
                    "message": f"Found {len(search_results)} results for: {search_query}",
                    "data": {
                        "query": search_query,
                        "results": search_results[:5],  # Limit results
                        "total_results": len(search_results)
                    }
                }
            except Exception as e:
                return {
                    "action": "development_error",
                    "message": f"Error searching web: {str(e)}",
                    "data": {"error": str(e)}
                }
        else:
            return {
                "action": "development_unavailable",
                "message": "Web search functionality is not available",
                "data": {}
            }

    def _handle_development_test(self, command: Command) -> Dict[str, Any]:
        """Handle testing command"""
        test_target = ""
        for entity in command.entities:
            if entity["name"] == "test_target":
                test_target = entity["value"]
                break

        if self.jarvis.tester:
            try:
                # Run tests for the target
                test_results = self.jarvis.tester.run_tests(
                    test_type="unit",
                    target=test_target,
                    comprehensive=True
                )

                return {
                    "action": "development_test",
                    "message": f"Testing completed for: {test_target}",
                    "data": {
                        "target": test_target,
                        "results": test_results,
                        "passed": test_results.get("success", False)
                    }
                }
            except Exception as e:
                return {
                    "action": "development_error",
                    "message": f"Error testing: {str(e)}",
                    "data": {"error": str(e)}
                }
        else:
            return {
                "action": "development_unavailable",
                "message": "Testing functionality is not available",
                "data": {}
            }

    def _handle_development_validate(self, command: Command) -> Dict[str, Any]:
        """Handle validation command with real code loading"""
        validation_target = ""
        for entity in command.entities:
            if entity["name"] == "validation_target":
                validation_target = entity["value"]
                break

        if not validation_target:
            return {
                "action": "development_error",
                "message": "No validation target specified",
                "data": {"error": "missing_target"}
            }

        if self.jarvis.self_development_engine and hasattr(self.jarvis.self_development_engine, 'validator'):
            try:
                # Load actual code based on the validation target
                code_to_validate = self._load_code_for_validation(validation_target)

                if not code_to_validate:
                    return {
                        "action": "development_error",
                        "message": f"Could not load code for validation target: {validation_target}",
                        "data": {"error": "code_not_found", "target": validation_target}
                    }

                # Validate the loaded code
                validation_results = self.jarvis.self_development_engine.validator.validate_code(code_to_validate)

                return {
                    "action": "development_validate",
                    "message": f"Validation completed for: {validation_target}",
                    "data": {
                        "target": validation_target,
                        "code_loaded": len(code_to_validate),
                        "results": validation_results,
                        "score": validation_results.get("overall_score", 0),
                        "issues_found": len(validation_results.get("issues", []))
                    }
                }
            except Exception as e:
                return {
                    "action": "development_error",
                    "message": f"Error validating: {str(e)}",
                    "data": {"error": str(e), "target": validation_target}
                }
        else:
            return {
                "action": "development_unavailable",
                "message": "Validation functionality is not available",
                "data": {}
            }

    def _load_code_for_validation(self, validation_target: str) -> Optional[str]:
        """Load actual code for validation based on target specification"""
        try:
            # Handle different types of validation targets

            # 1. Direct file path
            if validation_target.endswith('.py') and os.path.isfile(validation_target):
                with open(validation_target, 'r', encoding='utf-8') as f:
                    return f.read()

            # 2. Module name (e.g., "jarvis.core.command_processor")
            if '.' in validation_target and not validation_target.endswith('.py'):
                try:
                    module = __import__(validation_target, fromlist=[''])
                    module_file = getattr(module, '__file__', None)
                    if module_file and os.path.isfile(module_file):
                        with open(module_file, 'r', encoding='utf-8') as f:
                            return f.read()
                except ImportError:
                    pass

            # 3. Class or function name (try to find in current modules)
            # Search in common JARVIS modules
            search_paths = [
                'jarvis/core',
                'jarvis/modules',
                'jarvis/core/advanced',
                'jarvis/core/advanced/healer_components'
            ]

            for search_path in search_paths:
                if os.path.exists(search_path):
                    for root, dirs, files in os.walk(search_path):
                        for file in files:
                            if file.endswith('.py'):
                                file_path = os.path.join(root, file)
                                try:
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        content = f.read()

                                    # Check if the target is defined in this file
                                    if self._contains_definition(content, validation_target):
                                        return content

                                except (IOError, UnicodeDecodeError):
                                    continue

            # 4. Search by pattern in current codebase
            # Look for files that might contain the validation target
            jarvis_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for root, dirs, files in os.walk(jarvis_root):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()

                            # Check for class, function, or module definitions
                            if validation_target in content:
                                # More specific check - look for actual definitions
                                lines = content.split('\n')
                                for line in lines:
                                    line = line.strip()
                                    if (line.startswith('class ') and validation_target in line) or \
                                       (line.startswith('def ') and validation_target in line) or \
                                       (line.startswith('async def ') and validation_target in line):
                                        return content

                        except (IOError, UnicodeDecodeError):
                            continue

            # 5. If still not found, try to interpret as a code snippet
            # Check if it's actual Python code
            try:
                compile(validation_target, '<validation_target>', 'exec')
                return validation_target
            except SyntaxError:
                pass

            return None

        except Exception as e:
            self.jarvis.logger.warning(f"Error loading code for validation: {e}")
            return None

    def _contains_definition(self, content: str, target: str) -> bool:
        """Check if content contains a definition for the target"""
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            # Check for class definitions
            if line.startswith('class ') and target in line.split('(')[0]:
                return True
            # Check for function definitions
            if (line.startswith('def ') or line.startswith('async def ')) and target in line.split('(')[0]:
                return True
            # Check for module-level assignments
            if line.startswith(target + ' =') or line.startswith(target + '('):
                return True

        return False

    def _handle_development_reason(self, command: Command) -> Dict[str, Any]:
        """Handle reasoning command"""
        reasoning_topic = ""
        for entity in command.entities:
            if entity["name"] == "reasoning_topic":
                reasoning_topic = entity["value"]
                break

        if self.jarvis.self_development_engine and hasattr(self.jarvis.self_development_engine, 'reasoning_engine'):
            try:
                # Perform reasoning
                reasoning_results = self.jarvis.self_development_engine.reasoning_engine.reason(
                    task_description=f"Analyze and reason about: {reasoning_topic}",
                    research_data=[],
                    requirements={}
                )

                return {
                    "action": "development_reason",
                    "message": f"Reasoning completed for: {reasoning_topic}",
                    "data": {
                        "topic": reasoning_topic,
                        "results": reasoning_results,
                        "plan": reasoning_results.get("implementation_plan", "")
                    }
                }
            except Exception as e:
                return {
                    "action": "development_error",
                    "message": f"Error reasoning: {str(e)}",
                    "data": {"error": str(e)}
                }
        else:
            return {
                "action": "development_unavailable",
                "message": "Reasoning functionality is not available",
                "data": {}
            }

    def _handle_api_add_key(self, command: Command) -> Dict[str, Any]:
        """Handle API key addition command"""
        api_key = ""
        for entity in command.entities:
            if entity["name"] == "api_key":
                api_key = entity["value"]
                break

        if not api_key:
            return {
                "action": "api_error",
                "message": "No API key provided",
                "data": {"error": "missing_api_key"}
            }

        if not hasattr(self.jarvis, 'api_manager') or not self.jarvis.api_manager:
            return {
                "action": "api_error",
                "message": "API manager is not available",
                "data": {"error": "api_manager_unavailable"}
            }

        # Add API key asynchronously
        async def add_key_async():
            return await self.jarvis.api_manager.add_api_key(api_key)

        try:
            # Run in event loop if available, otherwise create one
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is already running, we need to handle this differently
                    # For now, return a message that the key will be processed
                    return {
                        "action": "api_add_key_queued",
                        "message": "API key addition queued for processing. Please wait...",
                        "data": {"api_key_preview": api_key[:10] + "..." if len(api_key) > 10 else api_key}
                    }
                else:
                    result = loop.run_until_complete(add_key_async())
                    return result
            except RuntimeError:
                # No event loop, create one
                result = asyncio.run(add_key_async())
                return result

        except Exception as e:
            return {
                "action": "api_error",
                "message": f"Error adding API key: {str(e)}",
                "data": {"error": str(e)}
            }

    def _handle_api_test_key(self, command: Command) -> Dict[str, Any]:
        """Handle API key testing command"""
        api_key = ""
        for entity in command.entities:
            if entity["name"] == "api_key":
                api_key = entity["value"]
                break

        if not api_key:
            return {
                "action": "api_error",
                "message": "No API key provided",
                "data": {"error": "missing_api_key"}
            }

        if not hasattr(self.jarvis, 'api_manager') or not self.jarvis.api_manager:
            return {
                "action": "api_error",
                "message": "API manager is not available",
                "data": {"error": "api_manager_unavailable"}
            }

        # Test API key asynchronously
        async def test_key_async():
            return await self.jarvis.api_manager.test_api_key(api_key)

        try:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    return {
                        "action": "api_test_key_queued",
                        "message": "API key testing queued for processing. Please wait...",
                        "data": {"api_key_preview": api_key[:10] + "..." if len(api_key) > 10 else api_key}
                    }
                else:
                    result = loop.run_until_complete(test_key_async())
                    return result
            except RuntimeError:
                result = asyncio.run(test_key_async())
                return result

        except Exception as e:
            return {
                "action": "api_error",
                "message": f"Error testing API key: {str(e)}",
                "data": {"error": str(e)}
            }

    def _handle_api_list_providers(self, command: Command) -> Dict[str, Any]:
        """Handle list API providers command"""
        if not hasattr(self.jarvis, 'api_manager') or not self.jarvis.api_manager:
            return {
                "action": "api_error",
                "message": "API manager is not available",
                "data": {"error": "api_manager_unavailable"}
            }

        try:
            providers = self.jarvis.api_manager.list_supported_providers()
            configured_providers = [p for p in providers if p.get("configured", False)]

            return {
                "action": "api_list_providers",
                "message": f"Found {len(providers)} supported providers, {len(configured_providers)} configured",
                "data": {
                    "total_providers": len(providers),
                    "configured_providers": len(configured_providers),
                    "providers": providers,
                    "configured_list": [p["name"] for p in configured_providers]
                }
            }

        except Exception as e:
            return {
                "action": "api_error",
                "message": f"Error listing providers: {str(e)}",
                "data": {"error": str(e)}
            }

    def _handle_api_remove_provider(self, command: Command) -> Dict[str, Any]:
        """Handle API provider removal command"""
        provider_name = ""
        for entity in command.entities:
            if entity["name"] == "provider_name":
                provider_name = entity["value"]
                break

        if not provider_name:
            return {
                "action": "api_error",
                "message": "No provider name provided",
                "data": {"error": "missing_provider_name"}
            }

        if not hasattr(self.jarvis, 'api_manager') or not self.jarvis.api_manager:
            return {
                "action": "api_error",
                "message": "API manager is not available",
                "data": {"error": "api_manager_unavailable"}
            }

        try:
            # Find provider by name
            from jarvis.core.api_manager import APIProvider
            target_provider = None

            for provider_enum in APIProvider:
                if provider_enum.value.lower() == provider_name.lower():
                    target_provider = provider_enum
                    break

            if not target_provider:
                return {
                    "action": "api_error",
                    "message": f"Provider '{provider_name}' not found",
                    "data": {"error": "provider_not_found", "provider_name": provider_name}
                }

            # Remove provider
            if target_provider in self.jarvis.api_manager.api_configs:
                del self.jarvis.api_manager.api_configs[target_provider]
                self.jarvis.api_manager.save_api_configurations()

                return {
                    "action": "api_remove_provider",
                    "message": f"Successfully removed API provider: {provider_name}",
                    "data": {"removed_provider": provider_name}
                }
            else:
                return {
                    "action": "api_error",
                    "message": f"Provider '{provider_name}' is not configured",
                    "data": {"error": "provider_not_configured", "provider_name": provider_name}
                }

        except Exception as e:
            return {
                "action": "api_error",
                "message": f"Error removing provider: {str(e)}",
                "data": {"error": str(e)}
            }

    def _handle_unknown_command(self, command: Command) -> Dict[str, Any]:
        """Handle unknown command"""
        # Try to suggest similar commands
        suggestions = self._get_command_suggestions(command.text)

        return {
            "action": "unknown_command",
            "message": "I'm not sure what you mean. Did you mean one of these?",
            "data": {
                "original_command": command.text,
                "suggestions": suggestions
            }
        }

    def _handle_unknown_intent(self, command: Command) -> Dict[str, Any]:
        """Handle unknown intent"""
        return {
            "action": "unknown_intent",
            "message": f"I don't know how to handle: {command.parsed_intent}",
            "data": {"intent": command.parsed_intent}
        }

    def _handle_command_error(self, command: Command, error: str) -> Dict[str, Any]:
        """Handle command execution error"""
        return {
            "action": "command_error",
            "message": f"Error executing command: {error}",
            "data": {"error": error}
        }

    def _get_command_suggestions(self, command_text: str) -> List[str]:
        """Get command suggestions based on input"""
        # Simple suggestion system based on history
        similar_commands = difflib.get_close_matches(
            command_text,
            [cmd["text"] for cmd in self.command_history[-50:]],
            n=3,
            cutoff=0.3
        )

        return similar_commands

    def _update_suggestions(self, command_text: str, intent: str):
        """Update command suggestions"""
        if command_text not in self.command_suggestions:
            self.command_suggestions[command_text] = {
                "intent": intent,
                "count": 1,
                "last_used": time.time()
            }
        else:
            self.command_suggestions[command_text]["count"] += 1
            self.command_suggestions[command_text]["last_used"] = time.time()

    def _add_to_history(self, command: Command):
        """Add command to history"""
        self.command_history.append(command.to_dict())

        # Maintain history size
        if len(self.command_history) > self.max_history_size:
            self.command_history.pop(0)

    def _update_processing_stats(self, success: bool, processing_time: float):
        """Update processing statistics"""
        self.processing_stats["total_commands"] += 1

        if success:
            self.processing_stats["successful_commands"] += 1
        else:
            self.processing_stats["failed_commands"] += 1

        # Update average processing time
        total_time = self.processing_stats["average_processing_time"] * (self.processing_stats["total_commands"] - 1)
        self.processing_stats["average_processing_time"] = (total_time + processing_time) / self.processing_stats["total_commands"]

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.processing_stats,
            "pending_commands": len(self.pending_commands),
            "processing_commands": len(self.processing_commands),
            "history_size": len(self.command_history),
            "registered_patterns": len(self.command_patterns),
            "registered_handlers": len(self.intent_handlers)
        }

    def clear_history(self):
        """Clear command history"""
        self.command_history.clear()
        self.command_suggestions.clear()
        self.jarvis.logger.info("Command history cleared")

    def get_recent_commands(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent commands from history"""
        return self.command_history[-limit:] if self.command_history else []

    def unregister_command_pattern(self, pattern: str, command_name: str) -> bool:
        """
        Unregister a command pattern

        Args:
            pattern: Regex pattern to remove
            command_name: Command name to remove

        Returns:
            Success status
        """
        try:
            # Find and remove the pattern
            intent_to_remove = None
            for intent, pattern_info in self.command_patterns.items():
                if pattern_info["pattern"].pattern == pattern:
                    intent_to_remove = intent
                    break

            if intent_to_remove:
                del self.command_patterns[intent_to_remove]
                if intent_to_remove in self.intent_handlers:
                    del self.intent_handlers[intent_to_remove]
                self.jarvis.logger.info(f"Unregistered command pattern: {command_name}")
                return True
            else:
                self.jarvis.logger.warning(f"Command pattern not found for: {command_name}")
                return False

        except Exception as e:
            self.jarvis.logger.error(f"Error unregistering command pattern: {e}")
            return False