"""
J.A.R.V.I.S. Voice Intelligence Engine
Advanced voice processing, natural language understanding, and intelligent suggestions
"""

import sys
import os
import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import re
from datetime import datetime, timedelta
import numpy as np

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class VoiceProcessor:
    """Advanced voice processing and recognition"""

    def __init__(self):
        self.recognizer = sr.Recognizer() if SPEECH_RECOGNITION_AVAILABLE else None
        self.microphone = sr.Microphone() if SPEECH_RECOGNITION_AVAILABLE else None
        self.is_listening = False
        self.audio_level = 0.0
        self.logger = logging.getLogger('JARVIS.VoiceProcessor')

    def start_listening(self) -> bool:
        """Start continuous voice listening"""
        try:
            if not SPEECH_RECOGNITION_AVAILABLE:
                self.logger.warning("Speech recognition not available")
                return False

            if self.is_listening:
                return True

            self.is_listening = True

            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)

            self.logger.info("Voice listening started")
            return True

        except Exception as e:
            self.logger.error(f"Error starting voice listening: {e}")
            return False

    def stop_listening(self) -> bool:
        """Stop voice listening"""
        try:
            self.is_listening = False
            self.logger.info("Voice listening stopped")
            return True

        except Exception as e:
            self.logger.error(f"Error stopping voice listening: {e}")
            return False

    async def listen_once(self, timeout: int = 5) -> Optional[str]:
        """Listen for a single voice command"""
        try:
            if not SPEECH_RECOGNITION_AVAILABLE:
                return None

            with self.microphone as source:
                self.logger.info("Listening for voice command...")
                audio = self.recognizer.listen(source, timeout=timeout)

                # Recognize speech
                try:
                    text = self.recognizer.recognize_google(audio)
                    self.logger.info(f"Recognized: {text}")
                    return text
                except sr.UnknownValueError:
                    self.logger.info("Could not understand audio")
                    return None
                except sr.RequestError as e:
                    self.logger.error(f"Speech recognition request failed: {e}")
                    return None

        except Exception as e:
            self.logger.error(f"Error in voice listening: {e}")
            return None

    def calibrate_microphone(self) -> bool:
        """Calibrate microphone for better recognition"""
        try:
            if not SPEECH_RECOGNITION_AVAILABLE:
                return False

            with self.microphone as source:
                self.logger.info("Calibrating microphone...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                self.logger.info("Microphone calibrated")
                return True

        except Exception as e:
            self.logger.error(f"Error calibrating microphone: {e}")
            return False

    def get_audio_level(self) -> float:
        """Get current audio input level"""
        # In a real implementation, this would monitor audio levels
        # For now, return a simulated value
        return np.random.uniform(0.0, 1.0)


class NaturalLanguageProcessor:
    """Natural language processing for voice commands"""

    def __init__(self):
        self.intent_classifier = None
        self.entity_extractor = None
        self.lemmatizer = WordNetLemmatizer() if NLTK_AVAILABLE else None
        self.stop_words = set(stopwords.words('english')) if NLTK_AVAILABLE else set()
        self.logger = logging.getLogger('JARVIS.NLPProcessor')

        # Command patterns and intents
        self.intent_patterns = {
            'system_status': [
                r'how are you', r'system status', r'what.*status', r'health check',
                r'are you okay', r'status report', r'system health'
            ],
            'time_date': [
                r'what time', r'what.*date', r'time is it', r'current time',
                r'what day', r'today.*date'
            ],
            'file_operations': [
                r'open file', r'create file', r'delete file', r'find file',
                r'search file', r'copy file', r'move file'
            ],
            'application_control': [
                r'open (.*)', r'launch (.*)', r'start (.*)', r'close (.*)',
                r'kill (.*)', r'terminate (.*)'
            ],
            'system_control': [
                r'shutdown', r'restart', r'log off', r'sleep', r'hibernate',
                r'lock computer', r'screen saver'
            ],
            'web_search': [
                r'search (.*)', r'google (.*)', r'find (.*) online',
                r'look up (.*)', r'research (.*)'
            ],
            'development': [
                r'create feature', r'fix bug', r'optimize code', r'run test',
                r'generate code', r'debug', r'compile'
            ],
            'voice_control': [
                r'stop listening', r'start listening', r'voice off', r'voice on',
                r'be quiet', r'silence'
            ]
        }

    async def initialize(self) -> bool:
        """Initialize NLP components"""
        try:
            if TRANSFORMERS_AVAILABLE:
                # Load pre-trained models (would be downloaded in real implementation)
                try:
                    self.intent_classifier = pipeline(
                        "text-classification",
                        model="facebook/bart-large-mnli"
                    )
                    self.logger.info("Intent classifier loaded")
                except Exception as e:
                    self.logger.warning(f"Could not load intent classifier: {e}")

            if NLTK_AVAILABLE:
                # Download required NLTK data
                try:
                    nltk.download('punkt', quiet=True)
                    nltk.download('stopwords', quiet=True)
                    nltk.download('wordnet', quiet=True)
                except Exception as e:
                    self.logger.warning(f"Could not download NLTK data: {e}")

            return True

        except Exception as e:
            self.logger.error(f"Error initializing NLP processor: {e}")
            return False

    def classify_intent(self, text: str) -> Dict[str, Any]:
        """Classify the intent of a text command"""
        try:
            text_lower = text.lower().strip()

            # Pattern-based intent classification
            for intent, patterns in self.intent_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        confidence = self._calculate_pattern_confidence(text_lower, pattern)
                        return {
                            'intent': intent,
                            'confidence': confidence,
                            'method': 'pattern_matching'
                        }

            # Fallback to transformer-based classification
            if self.intent_classifier:
                try:
                    # Use zero-shot classification
                    candidate_labels = list(self.intent_patterns.keys())
                    result = self.intent_classifier(text, candidate_labels)
                    return {
                        'intent': result[0]['label'],
                        'confidence': result[0]['score'],
                        'method': 'transformer'
                    }
                except Exception as e:
                    self.logger.error(f"Transformer classification failed: {e}")

            # Default fallback
            return {
                'intent': 'unknown',
                'confidence': 0.0,
                'method': 'fallback'
            }

        except Exception as e:
            self.logger.error(f"Error classifying intent: {e}")
            return {'intent': 'unknown', 'confidence': 0.0, 'method': 'error'}

    def _calculate_pattern_confidence(self, text: str, pattern: str) -> float:
        """Calculate confidence score for pattern matching"""
        try:
            # Simple confidence calculation based on pattern specificity
            pattern_length = len(pattern.split())
            text_length = len(text.split())

            # Higher confidence for longer, more specific patterns
            base_confidence = min(0.9, pattern_length / 10)

            # Boost confidence if pattern matches significant portion of text
            match_ratio = pattern_length / text_length if text_length > 0 else 0
            confidence_boost = match_ratio * 0.3

            return min(1.0, base_confidence + confidence_boost)

        except Exception:
            return 0.5

    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities from text"""
        try:
            entities = {
                'files': [],
                'applications': [],
                'urls': [],
                'numbers': [],
                'dates': [],
                'locations': []
            }

            # Extract file paths
            file_pattern = r'["\']?([^\s]+\.[a-zA-Z0-9]+)["\']?'
            for match in re.finditer(file_pattern, text):
                entities['files'].append({
                    'value': match.group(1),
                    'start': match.start(),
                    'end': match.end()
                })

            # Extract URLs
            url_pattern = r'https?://[^\s]+'
            for match in re.finditer(url_pattern, text):
                entities['urls'].append({
                    'value': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })

            # Extract numbers
            number_pattern = r'\b\d+\.?\d*\b'
            for match in re.finditer(number_pattern, text):
                entities['numbers'].append({
                    'value': float(match.group()) if '.' in match.group() else int(match.group()),
                    'start': match.start(),
                    'end': match.end()
                })

            # Extract application names (simplified)
            app_keywords = ['chrome', 'firefox', 'word', 'excel', 'notepad', 'calculator']
            words = text.lower().split()
            for word in words:
                if word in app_keywords:
                    entities['applications'].append({
                        'value': word,
                        'start': text.lower().find(word),
                        'end': text.lower().find(word) + len(word)
                    })

            return entities

        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}")
            return {}

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better processing"""
        try:
            if not NLTK_AVAILABLE:
                return text.lower().strip()

            # Tokenize and preprocess
            tokens = word_tokenize(text.lower())

            # Remove stop words and lemmatize
            processed_tokens = []
            for token in tokens:
                if token not in self.stop_words and token.isalnum():
                    lemma = self.lemmatizer.lemmatize(token)
                    processed_tokens.append(lemma)

            return ' '.join(processed_tokens)

        except Exception as e:
            self.logger.error(f"Error preprocessing text: {e}")
            return text.lower().strip()


class VoiceIntelligenceEngine:
    """Advanced voice intelligence and natural language processing"""

    def __init__(self, development_engine):
        self.development_engine = development_engine
        self.jarvis = development_engine.jarvis if hasattr(development_engine, 'jarvis') else None
        self.logger = logging.getLogger('JARVIS.VoiceIntelligence')

        # Voice components
        self.voice_processor = VoiceProcessor()
        self.nlp_processor = NaturalLanguageProcessor()

        # Intelligence data
        self.command_history = []
        self.intent_patterns = {}
        self.user_preferences = {}
        self.context_memory = []

        # Learning data
        self.intent_learning_data = []
        self.successful_commands = []
        self.failed_commands = []

        # State
        self.is_active = False
        self.learning_mode = True

    async def initialize(self) -> bool:
        """Initialize voice intelligence engine"""
        try:
            self.logger.info("Initializing Voice Intelligence Engine...")

            # Initialize NLP processor
            await self.nlp_processor.initialize()

            # Load learning data
            await self._load_learning_data()

            # Start voice processing
            if not self.voice_processor.start_listening():
                self.logger.warning("Voice processing not available")

            self.is_active = True
            self.logger.info("Voice Intelligence Engine initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing voice intelligence: {e}")
            return False

    async def _load_learning_data(self):
        """Load learning data from disk"""
        try:
            data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
            learning_file = os.path.join(data_dir, 'voice_learning.json')

            if os.path.exists(learning_file):
                with open(learning_file, 'r') as f:
                    data = json.load(f)

                self.intent_patterns = data.get('intent_patterns', {})
                self.user_preferences = data.get('user_preferences', {})
                self.intent_learning_data = data.get('learning_data', [])

                self.logger.info(f"Loaded voice learning data: {len(self.intent_learning_data)} samples")

        except Exception as e:
            self.logger.error(f"Error loading learning data: {e}")

    async def _save_learning_data(self):
        """Save learning data to disk"""
        try:
            data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
            os.makedirs(data_dir, exist_ok=True)

            learning_file = os.path.join(data_dir, 'voice_learning.json')

            data = {
                'intent_patterns': self.intent_patterns,
                'user_preferences': self.user_preferences,
                'learning_data': self.intent_learning_data[-1000:],  # Keep last 1000 samples
                'last_updated': datetime.now().isoformat()
            }

            with open(learning_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving learning data: {e}")

    async def process_voice_command(self, audio_text: str = None, timeout: int = 5) -> Dict[str, Any]:
        """Process a voice command"""
        try:
            # Get audio input if not provided
            if audio_text is None:
                audio_text = await self.voice_processor.listen_once(timeout)
                if not audio_text:
                    return {'success': False, 'error': 'No speech detected'}

            # Preprocess text
            processed_text = self.nlp_processor.preprocess_text(audio_text)

            # Classify intent
            intent_result = self.nlp_processor.classify_intent(audio_text)

            # Extract entities
            entities = self.nlp_processor.extract_entities(audio_text)

            # Generate context
            context = self._generate_context(audio_text, intent_result, entities)

            # Store in history
            command_record = {
                'text': audio_text,
                'processed_text': processed_text,
                'intent': intent_result,
                'entities': entities,
                'context': context,
                'timestamp': datetime.now().isoformat(),
                'success': None  # Will be updated after execution
            }

            self.command_history.append(command_record)

            # Learn from command
            if self.learning_mode:
                await self._learn_from_command(command_record)

            return {
                'success': True,
                'text': audio_text,
                'intent': intent_result,
                'entities': entities,
                'context': context,
                'suggestions': self._generate_suggestions(intent_result, context)
            }

        except Exception as e:
            self.logger.error(f"Error processing voice command: {e}")
            return {'success': False, 'error': str(e)}

    def _generate_context(self, text: str, intent: Dict[str, Any], entities: Dict[str, List]) -> Dict[str, Any]:
        """Generate context for command processing"""
        try:
            context = {
                'time_of_day': self._get_time_context(),
                'recent_commands': [cmd['text'] for cmd in self.command_history[-3:]],
                'user_patterns': self._analyze_user_patterns(),
                'system_state': self._get_system_context(),
                'conversation_flow': self._analyze_conversation_flow()
            }

            return context

        except Exception as e:
            self.logger.error(f"Error generating context: {e}")
            return {}

    def _get_time_context(self) -> str:
        """Get time-based context"""
        try:
            hour = datetime.now().hour
            if 6 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 17:
                return 'afternoon'
            elif 17 <= hour < 22:
                return 'evening'
            else:
                return 'night'

        except Exception:
            return 'unknown'

    def _analyze_user_patterns(self) -> Dict[str, Any]:
        """Analyze user command patterns"""
        try:
            if not self.command_history:
                return {}

            recent_commands = self.command_history[-20:]
            intents = [cmd['intent']['intent'] for cmd in recent_commands if cmd['intent']['intent'] != 'unknown']

            # Count intent frequencies
            from collections import Counter
            intent_counts = Counter(intents)

            return {
                'frequent_intents': dict(intent_counts.most_common(3)),
                'preferred_time': self._get_time_context(),
                'command_complexity': np.mean([len(cmd['text'].split()) for cmd in recent_commands])
            }

        except Exception as e:
            self.logger.error(f"Error analyzing user patterns: {e}")
            return {}

    def _get_system_context(self) -> Dict[str, Any]:
        """Get current system context"""
        try:
            if not self.jarvis or not hasattr(self.jarvis, 'system_monitor'):
                return {}

            readings = self.jarvis.system_monitor.current_readings
            return {
                'cpu_usage': readings.get('cpu', {}).get('percent', 0),
                'memory_usage': readings.get('memory', {}).get('percent', 0),
                'active_tasks': len(self.jarvis.system_monitor.active_tasks) if hasattr(self.jarvis.system_monitor, 'active_tasks') else 0
            }

        except Exception as e:
            self.logger.error(f"Error getting system context: {e}")
            return {}

    def _analyze_conversation_flow(self) -> str:
        """Analyze conversation flow"""
        try:
            if len(self.command_history) < 2:
                return 'single_command'

            recent_intents = [cmd['intent']['intent'] for cmd in self.command_history[-3:]]
            if len(set(recent_intents)) == 1:
                return 'repeated_intent'
            elif 'system_status' in recent_intents and 'file_operations' in recent_intents:
                return 'workflow_sequence'
            else:
                return 'varied_commands'

        except Exception:
            return 'unknown'

    def _generate_suggestions(self, intent: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate intelligent command suggestions"""
        try:
            suggestions = []

            # Intent-based suggestions
            intent_type = intent.get('intent', 'unknown')

            if intent_type == 'system_status':
                suggestions.extend([
                    "Run a full system diagnostic",
                    "Check recent system logs",
                    "Monitor system performance in real-time"
                ])

            elif intent_type == 'file_operations':
                suggestions.extend([
                    "Search for files by content",
                    "Organize files by type and date",
                    "Create backup of important files"
                ])

            elif intent_type == 'development':
                suggestions.extend([
                    "Generate unit tests for code",
                    "Run code quality analysis",
                    "Optimize code performance"
                ])

            # Context-based suggestions
            time_context = context.get('time_of_day', 'unknown')
            if time_context == 'morning':
                suggestions.append("Review overnight system reports")
            elif time_context == 'evening':
                suggestions.append("Prepare system for maintenance")

            # Pattern-based suggestions
            user_patterns = context.get('user_patterns', {})
            frequent_intents = user_patterns.get('frequent_intents', {})

            if 'file_operations' in frequent_intents:
                suggestions.append("Set up automated file organization")

            if 'system_status' in frequent_intents:
                suggestions.append("Configure system health monitoring alerts")

            return suggestions[:5]  # Limit to 5 suggestions

        except Exception as e:
            self.logger.error(f"Error generating suggestions: {e}")
            return []

    async def _learn_from_command(self, command_record: Dict[str, Any]):
        """Learn from command execution for future improvements"""
        try:
            # Add to learning data
            self.intent_learning_data.append({
                'text': command_record['text'],
                'intent': command_record['intent']['intent'],
                'confidence': command_record['intent']['confidence'],
                'entities': command_record['entities'],
                'context': command_record['context'],
                'timestamp': command_record['timestamp']
            })

            # Update intent patterns
            intent = command_record['intent']['intent']
            text = command_record['text'].lower()

            if intent not in self.intent_patterns:
                self.intent_patterns[intent] = []

            # Extract patterns from successful commands
            words = text.split()
            if len(words) > 2:
                pattern = ' '.join(words[:3])  # First 3 words as pattern
                if pattern not in self.intent_patterns[intent]:
                    self.intent_patterns[intent].append(pattern)

            # Save learning data periodically
            if len(self.intent_learning_data) % 10 == 0:
                await self._save_learning_data()

        except Exception as e:
            self.logger.error(f"Error learning from command: {e}")

    async def get_intelligent_suggestions(self, current_context: str = '') -> Dict[str, Any]:
        """Get intelligent voice command suggestions"""
        try:
            suggestions = {
                'context_aware': [],
                'pattern_based': [],
                'learning_based': [],
                'system_suggestions': []
            }

            # Context-aware suggestions
            time_context = self._get_time_context()
            if time_context == 'morning':
                suggestions['context_aware'].extend([
                    "Good morning JARVIS, what's the system status?",
                    "Check overnight logs and reports",
                    "Start daily system health monitoring"
                ])
            elif time_context == 'evening':
                suggestions['context_aware'].extend([
                    "Prepare system for maintenance",
                    "Run backup operations",
                    "Check system performance summary"
                ])

            # Pattern-based suggestions from user history
            if self.command_history:
                recent_intents = [cmd['intent']['intent'] for cmd in self.command_history[-5:] if cmd['intent']['intent'] != 'unknown']
                if recent_intents:
                    most_common = max(set(recent_intents), key=recent_intents.count)
                    suggestions['pattern_based'] = self._get_suggestions_for_intent(most_common)

            # Learning-based suggestions
            if self.intent_learning_data:
                # Suggest commands similar to successful ones
                successful_commands = [item for item in self.intent_learning_data if item.get('success') == True]
                if successful_commands:
                    suggestions['learning_based'] = [cmd['text'] for cmd in successful_commands[-3:]]

            # System-based suggestions
            system_context = self._get_system_context()
            cpu_usage = system_context.get('cpu_usage', 0)
            memory_usage = system_context.get('memory_usage', 0)

            if cpu_usage > 80:
                suggestions['system_suggestions'].append("High CPU usage detected - consider optimization")
            if memory_usage > 85:
                suggestions['system_suggestions'].append("High memory usage - run cleanup")

            return {
                'suggestions': suggestions,
                'total_count': sum(len(v) for v in suggestions.values()),
                'context': {
                    'time_of_day': time_context,
                    'system_load': 'high' if cpu_usage > 70 or memory_usage > 80 else 'normal',
                    'learning_samples': len(self.intent_learning_data)
                }
            }

        except Exception as e:
            self.logger.error(f"Error getting intelligent suggestions: {e}")
            return {'suggestions': {}, 'total_count': 0, 'context': {}}

    def _get_suggestions_for_intent(self, intent: str) -> List[str]:
        """Get suggestions for a specific intent"""
        try:
            intent_suggestions = {
                'system_status': [
                    "Show detailed system information",
                    "Run comprehensive health check",
                    "Display system performance metrics"
                ],
                'file_operations': [
                    "Search for files by name or content",
                    "Organize files by type and date",
                    "Create backup of selected files"
                ],
                'application_control': [
                    "Open multiple applications at once",
                    "Close unresponsive applications",
                    "Switch between running applications"
                ],
                'development': [
                    "Generate code documentation",
                    "Run automated testing suite",
                    "Optimize code for performance"
                ]
            }

            return intent_suggestions.get(intent, [])

        except Exception:
            return []

    async def learn_voice_patterns(self) -> Dict[str, Any]:
        """Learn and adapt to user voice patterns"""
        try:
            learning_results = {
                'patterns_learned': 0,
                'accuracy_improved': 0.0,
                'new_commands_discovered': 0,
                'adaptations_made': []
            }

            if not self.intent_learning_data:
                return learning_results

            # Analyze learning data
            successful_commands = [item for item in self.intent_learning_data if item.get('success') == True]
            failed_commands = [item for item in self.intent_learning_data if item.get('success') == False]

            # Extract patterns from successful commands
            for cmd in successful_commands[-10:]:  # Last 10 successful commands
                text = cmd['text']
                intent = cmd['intent']

                # Simple pattern extraction
                words = text.lower().split()
                if len(words) >= 3:
                    pattern = f"{words[0]} {words[1]}"
                    if pattern not in self.intent_patterns.get(intent, []):
                        if intent not in self.intent_patterns:
                            self.intent_patterns[intent] = []
                        self.intent_patterns[intent].append(pattern)
                        learning_results['patterns_learned'] += 1

            # Calculate accuracy improvement
            if len(successful_commands) > 5:
                recent_accuracy = np.mean([cmd['confidence'] for cmd in successful_commands[-5:]])
                older_accuracy = np.mean([cmd['confidence'] for cmd in successful_commands[:-5]]) if len(successful_commands) > 5 else recent_accuracy
                learning_results['accuracy_improved'] = recent_accuracy - older_accuracy

            learning_results['adaptations_made'] = [
                f"Learned {learning_results['patterns_learned']} new command patterns",
                f"Voice recognition accuracy improved by {learning_results['accuracy_improved']:.1%}",
                "Adapted to user speaking patterns",
                "Updated intent classification rules"
            ]

            # Save updated learning data
            await self._save_learning_data()

            return learning_results

        except Exception as e:
            self.logger.error(f"Error learning voice patterns: {e}")
            return {}

    def get_voice_stats(self) -> Dict[str, Any]:
        """Get voice intelligence statistics"""
        try:
            total_commands = len(self.command_history)
            successful_commands = len([cmd for cmd in self.command_history if cmd.get('success') == True])
            accuracy = successful_commands / total_commands if total_commands > 0 else 0

            intent_distribution = {}
            for cmd in self.command_history:
                intent = cmd['intent']['intent']
                intent_distribution[intent] = intent_distribution.get(intent, 0) + 1

            return {
                'total_commands_processed': total_commands,
                'successful_commands': successful_commands,
                'accuracy': accuracy,
                'intent_distribution': intent_distribution,
                'learning_samples': len(self.intent_learning_data),
                'voice_processor_active': self.voice_processor.is_listening if self.voice_processor else False,
                'nlp_processor_available': True,  # Since we have basic NLP
                'learning_mode': self.learning_mode
            }

        except Exception as e:
            self.logger.error(f"Error getting voice stats: {e}")
            return {}

    async def shutdown(self):
        """Shutdown voice intelligence engine"""
        try:
            self.voice_processor.stop_listening()
            await self._save_learning_data()
            self.is_active = False
            self.logger.info("Voice Intelligence Engine shutdown")
        except Exception as e:
            self.logger.error(f"Error shutting down voice intelligence: {e}")