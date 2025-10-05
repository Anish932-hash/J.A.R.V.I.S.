"""
J.A.R.V.I.S. Reasoning Engine
AI-powered reasoning and planning system
"""

import os
import time
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import re


class KnowledgeGraph:
    """Knowledge graph for reasoning"""

    def __init__(self):
        self.nodes = {}  # concept -> properties
        self.edges = defaultdict(list)  # concept -> [(relation, target), ...]

    def add_concept(self, concept: str, properties: Dict[str, Any] = None):
        """Add concept to knowledge graph"""
        if concept not in self.nodes:
            self.nodes[concept] = properties or {}
        else:
            self.nodes[concept].update(properties or {})

    def add_relation(self, source: str, relation: str, target: str, properties: Dict[str, Any] = None):
        """Add relation between concepts"""
        self.edges[source].append((relation, target, properties or {}))

    def get_related_concepts(self, concept: str, relation: str = None) -> List[str]:
        """Get concepts related to given concept"""
        related = []
        for rel, target, _ in self.edges[concept]:
            if relation is None or rel == relation:
                related.append(target)
        return related

    def find_path(self, start: str, end: str, max_depth: int = 3) -> List[List[str]]:
        """Find paths between concepts"""
        paths = []

        def dfs(current: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            if current == end:
                paths.append(path[:])
                return

            for _, neighbor, _ in self.edges[current]:
                if neighbor not in path:
                    path.append(neighbor)
                    dfs(neighbor, path, depth + 1)
                    path.pop()

        dfs(start, [start], 0)
        return paths


class ReasoningEngine:
    """AI-powered reasoning for development tasks"""

    def __init__(self, development_engine):
        self.engine = development_engine
        self.jarvis = development_engine.jarvis
        self.logger = logging.getLogger('JARVIS.ReasoningEngine')

        # Knowledge base
        self.knowledge_graph = KnowledgeGraph()
        self.patterns_db = {}
        self.solution_templates = {}

        # Reasoning state
        self.reasoning_history = []

    async def initialize(self):
        """Initialize reasoning engine"""
        try:
            self.logger.info("Initializing reasoning engine...")

            # Load knowledge base
            await self._load_knowledge_base()

            # Initialize patterns and templates
            await self._initialize_patterns()

            self.logger.info("Reasoning engine initialized")

        except Exception as e:
            self.logger.error(f"Error initializing reasoning engine: {e}")
            raise

    async def _load_knowledge_base(self):
        """Load knowledge base from files"""
        try:
            kb_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'knowledge_base.json')

            if os.path.exists(kb_file):
                with open(kb_file, 'r') as f:
                    kb_data = json.load(f)

                # Load concepts
                for concept, properties in kb_data.get('concepts', {}).items():
                    self.knowledge_graph.add_concept(concept, properties)

                # Load relations
                for source, relations in kb_data.get('relations', {}).items():
                    for relation_data in relations:
                        self.knowledge_graph.add_relation(
                            source,
                            relation_data['relation'],
                            relation_data['target'],
                            relation_data.get('properties', {})
                        )

            else:
                # Initialize with basic programming knowledge
                await self._initialize_basic_knowledge()

        except Exception as e:
            self.logger.error(f"Error loading knowledge base: {e}")

    async def _initialize_basic_knowledge(self):
        """Initialize basic programming knowledge"""
        # Programming concepts
        concepts = [
            ("python", {"type": "language", "paradigm": ["object-oriented", "functional"]}),
            ("error_handling", {"type": "concept", "importance": "high"}),
            ("logging", {"type": "library", "purpose": "debugging"}),
            ("async", {"type": "feature", "complexity": "medium"}),
            ("testing", {"type": "practice", "importance": "high"}),
            ("performance", {"type": "concern", "metrics": ["speed", "memory"]}),
            ("security", {"type": "concern", "aspects": ["authentication", "encryption"]}),
        ]

        for concept, properties in concepts:
            self.knowledge_graph.add_concept(concept, properties)

        # Relations
        relations = [
            ("python", "has_feature", "error_handling"),
            ("python", "has_feature", "async"),
            ("error_handling", "requires", "try_except"),
            ("logging", "helps_with", "debugging"),
            ("testing", "ensures", "reliability"),
            ("performance", "affects", "user_experience"),
            ("security", "prevents", "vulnerabilities"),
        ]

        for source, relation, target in relations:
            self.knowledge_graph.add_relation(source, relation, target)

    async def _initialize_patterns(self):
        """Initialize reasoning patterns"""
        self.patterns_db = {
            "web_api": {
                "indicators": ["api", "http", "request", "response", "endpoint"],
                "requirements": ["requests", "aiohttp", "error_handling"],
                "complexity": "medium"
            },
            "data_processing": {
                "indicators": ["data", "process", "transform", "analyze", "csv", "json"],
                "requirements": ["pandas", "numpy", "data_validation"],
                "complexity": "medium"
            },
            "file_operation": {
                "indicators": ["file", "read", "write", "directory", "path"],
                "requirements": ["os", "pathlib", "error_handling"],
                "complexity": "low"
            },
            "machine_learning": {
                "indicators": ["model", "predict", "train", "dataset", "algorithm"],
                "requirements": ["scikit-learn", "tensorflow", "data_preprocessing"],
                "complexity": "high"
            },
            "gui_application": {
                "indicators": ["interface", "window", "button", "display", "ui"],
                "requirements": ["tkinter", "PyQt", "event_handling"],
                "complexity": "medium"
            }
        }

    async def reason(self, task_description: str, research_data: List[Dict[str, Any]], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Reason about implementation approach"""
        try:
            self.logger.info(f"Starting reasoning for: {task_description}")

            reasoning_start = time.time()

            # Analyze task description
            task_analysis = self._analyze_task_description(task_description)

            # Analyze research data
            research_insights = self._analyze_research_data(research_data)

            # Identify patterns and requirements
            identified_patterns = self._identify_patterns(task_description, research_data)

            # Generate implementation plan
            implementation_plan = self._generate_implementation_plan(
                task_analysis, research_insights, identified_patterns, requirements
            )

            # Determine technical requirements
            technical_requirements = self._determine_technical_requirements(
                identified_patterns, requirements
            )

            # Identify dependencies
            dependencies = self._identify_dependencies(technical_requirements, identified_patterns)

            # Risk assessment
            risks = self._assess_risks(implementation_plan, technical_requirements)

            reasoning_time = time.time() - reasoning_start

            # Record reasoning
            reasoning_record = {
                "task_description": task_description,
                "task_analysis": task_analysis,
                "research_insights": research_insights,
                "patterns": identified_patterns,
                "implementation_plan": implementation_plan,
                "technical_requirements": technical_requirements,
                "dependencies": dependencies,
                "risks": risks,
                "reasoning_time": reasoning_time,
                "timestamp": time.time()
            }

            self.reasoning_history.append(reasoning_record)

            result = {
                "success": True,
                "implementation_plan": implementation_plan,
                "technical_requirements": technical_requirements,
                "dependencies": dependencies,
                "risks": risks,
                "confidence": self._calculate_confidence(reasoning_record),
                "reasoning_time": reasoning_time
            }

            self.logger.info(f"Reasoning completed in {reasoning_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error in reasoning: {e}")
            return {
                "success": False,
                "error": str(e),
                "implementation_plan": f"Basic plan for {task_description}",
                "technical_requirements": ["Python"],
                "dependencies": []
            }

    def _analyze_task_description(self, description: str) -> Dict[str, Any]:
        """Analyze task description for key components"""
        analysis = {
            "keywords": [],
            "task_type": "unknown",
            "complexity": "medium",
            "domain": "general"
        }

        # Extract keywords
        words = re.findall(r'\b\w+\b', description.lower())
        analysis["keywords"] = [word for word in words if len(word) > 2]

        # Determine task type
        if any(word in description.lower() for word in ["create", "build", "implement", "develop"]):
            analysis["task_type"] = "development"
        elif any(word in description.lower() for word in ["fix", "repair", "resolve", "correct"]):
            analysis["task_type"] = "bug_fix"
        elif any(word in description.lower() for word in ["optimize", "improve", "speed up"]):
            analysis["task_type"] = "optimization"
        elif any(word in description.lower() for word in ["test", "verify", "validate"]):
            analysis["task_type"] = "testing"

        # Determine domain
        if any(word in description.lower() for word in ["web", "api", "http", "server"]):
            analysis["domain"] = "web"
        elif any(word in description.lower() for word in ["data", "database", "sql", "analysis"]):
            analysis["domain"] = "data"
        elif any(word in description.lower() for word in ["file", "directory", "filesystem"]):
            analysis["domain"] = "filesystem"
        elif any(word in description.lower() for word in ["ai", "ml", "machine learning", "neural"]):
            analysis["domain"] = "ai"

        # Estimate complexity
        complexity_indicators = ["complex", "advanced", "sophisticated", "multiple", "integration"]
        if any(indicator in description.lower() for indicator in complexity_indicators):
            analysis["complexity"] = "high"
        elif any(word in description.lower() for word in ["simple", "basic", "straightforward"]):
            analysis["complexity"] = "low"

        return analysis

    def _analyze_research_data(self, research_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze research data for insights"""
        insights = {
            "key_concepts": [],
            "technologies": [],
            "patterns": [],
            "challenges": [],
            "solutions": []
        }

        if not research_data:
            return insights

        # Extract information from research data
        all_text = ""
        for item in research_data:
            if isinstance(item, dict):
                if "content" in item:
                    all_text += item["content"] + " "
                elif "snippet" in item:
                    all_text += item["snippet"] + " "
                elif "description" in item:
                    all_text += item["description"] + " "

        # Extract key concepts (simplified NLP)
        words = re.findall(r'\b\w+\b', all_text.lower())
        word_freq = Counter(words)

        # Filter common programming terms
        programming_terms = [
            "python", "function", "class", "method", "variable", "error", "exception",
            "api", "database", "file", "network", "security", "performance", "testing"
        ]

        insights["key_concepts"] = [term for term in programming_terms if term in word_freq]

        # Extract technologies
        technologies = [
            "django", "flask", "fastapi", "tensorflow", "pytorch", "pandas", "numpy",
            "postgresql", "mysql", "mongodb", "redis", "docker", "kubernetes"
        ]

        insights["technologies"] = [tech for tech in technologies if tech in all_text.lower()]

        # Identify patterns
        pattern_indicators = {
            "async_await": ["async", "await", "asyncio"],
            "error_handling": ["try", "except", "finally"],
            "logging": ["logger", "logging"],
            "testing": ["pytest", "unittest", "test"],
            "web_framework": ["flask", "django", "fastapi"]
        }

        for pattern, indicators in pattern_indicators.items():
            if any(indicator in all_text.lower() for indicator in indicators):
                insights["patterns"].append(pattern)

        return insights

    def _identify_patterns(self, task_description: str, research_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify relevant patterns for the task"""
        identified_patterns = []

        # Combine description and research data
        combined_text = task_description.lower()
        for item in research_data:
            if isinstance(item, dict):
                for key in ["content", "snippet", "description"]:
                    if key in item:
                        combined_text += " " + str(item[key]).lower()

        # Check against pattern database
        for pattern_name, pattern_data in self.patterns_db.items():
            indicators = pattern_data["indicators"]
            matches = sum(1 for indicator in indicators if indicator in combined_text)

            if matches >= len(indicators) * 0.5:  # 50% match threshold
                identified_patterns.append({
                    "pattern": pattern_name,
                    "confidence": matches / len(indicators),
                    "requirements": pattern_data["requirements"],
                    "complexity": pattern_data["complexity"]
                })

        # Sort by confidence
        identified_patterns.sort(key=lambda x: x["confidence"], reverse=True)

        return identified_patterns

    def _generate_implementation_plan(self, task_analysis: Dict[str, Any],
                                   research_insights: Dict[str, Any],
                                   patterns: List[Dict[str, Any]],
                                   requirements: Dict[str, Any]) -> str:
        """Generate detailed implementation plan"""
        plan_parts = []

        # Introduction
        plan_parts.append(f"Implementation Plan for: {task_analysis.get('task_type', 'Unknown').title()} Task")
        plan_parts.append("")

        # Task overview
        plan_parts.append("Task Overview:")
        plan_parts.append(f"- Type: {task_analysis.get('task_type', 'unknown')}")
        plan_parts.append(f"- Domain: {task_analysis.get('domain', 'general')}")
        plan_parts.append(f"- Complexity: {task_analysis.get('complexity', 'medium')}")
        plan_parts.append("")

        # Key insights from research
        if research_insights.get("key_concepts"):
            plan_parts.append("Key Concepts:")
            for concept in research_insights["key_concepts"]:
                plan_parts.append(f"- {concept}")
            plan_parts.append("")

        if research_insights.get("technologies"):
            plan_parts.append("Recommended Technologies:")
            for tech in research_insights["technologies"]:
                plan_parts.append(f"- {tech}")
            plan_parts.append("")

        # Implementation steps
        plan_parts.append("Implementation Steps:")

        if patterns:
            primary_pattern = patterns[0]
            plan_parts.append(f"1. Implement core {primary_pattern['pattern']} functionality")
            plan_parts.append(f"2. Add error handling and logging")
            plan_parts.append(f"3. Implement testing and validation")
            plan_parts.append(f"4. Add performance optimizations")
            plan_parts.append(f"5. Documentation and deployment")
        else:
            plan_parts.append("1. Analyze requirements and design solution")
            plan_parts.append("2. Implement core functionality")
            plan_parts.append("3. Add error handling and edge cases")
            plan_parts.append("4. Testing and validation")
            plan_parts.append("5. Optimization and documentation")

        plan_parts.append("")

        # Risk mitigation
        plan_parts.append("Risk Mitigation:")
        plan_parts.append("- Implement comprehensive error handling")
        plan_parts.append("- Add logging for debugging")
        plan_parts.append("- Write unit tests for critical functions")
        plan_parts.append("- Consider performance implications")

        return "\n".join(plan_parts)

    def _determine_technical_requirements(self, patterns: List[Dict[str, Any]],
                                        requirements: Dict[str, Any]) -> List[str]:
        """Determine technical requirements"""
        tech_requirements = set()

        # From patterns
        for pattern in patterns:
            for req in pattern.get("requirements", []):
                tech_requirements.add(req)

        # From explicit requirements
        for key, value in requirements.items():
            if isinstance(value, list):
                tech_requirements.update(value)
            elif isinstance(value, str):
                tech_requirements.add(value)

        # Add common requirements
        tech_requirements.update([
            "Python 3.8+",
            "Error handling",
            "Logging",
            "Type hints",
            "Documentation"
        ])

        return sorted(list(tech_requirements))

    def _identify_dependencies(self, technical_requirements: List[str],
                             patterns: List[Dict[str, Any]]) -> List[str]:
        """Identify required dependencies"""
        dependencies = set()

        # Map requirements to packages
        package_mapping = {
            "requests": ["requests"],
            "aiohttp": ["aiohttp"],
            "pandas": ["pandas", "numpy"],
            "numpy": ["numpy"],
            "scikit-learn": ["scikit-learn"],
            "tensorflow": ["tensorflow"],
            "PyQt": ["PyQt6"],
            "tkinter": ["tkinter"],  # Built-in
            "pytest": ["pytest"],
            "logging": [],  # Built-in
            "os": [],  # Built-in
            "pathlib": [],  # Built-in
        }

        for req in technical_requirements:
            req_lower = req.lower()
            for key, packages in package_mapping.items():
                if key in req_lower:
                    dependencies.update(packages)

        # Add common dependencies
        dependencies.update(["asyncio", "typing", "json", "os", "sys"])

        return sorted(list(dependencies))

    def _assess_risks(self, implementation_plan: str, technical_requirements: List[str]) -> List[str]:
        """Assess implementation risks"""
        risks = []

        # Complexity risks
        if "high" in implementation_plan.lower():
            risks.append("High complexity may require additional testing")

        # Technology risks
        new_technologies = [req for req in technical_requirements
                          if req in ["tensorflow", "kubernetes", "machine learning"]]
        if new_technologies:
            risks.append(f"New technologies ({', '.join(new_technologies)}) may require learning curve")

        # Dependency risks
        if len(technical_requirements) > 10:
            risks.append("Large number of dependencies increases maintenance complexity")

        # Performance risks
        if "performance" in implementation_plan.lower():
            risks.append("Performance-critical features need thorough benchmarking")

        # Security risks
        if any(word in implementation_plan.lower() for word in ["security", "authentication", "encryption"]):
            risks.append("Security features require security audit")

        return risks

    def _calculate_confidence(self, reasoning_record: Dict[str, Any]) -> float:
        """Calculate confidence in reasoning results"""
        confidence = 0.5  # Base confidence

        # Boost confidence based on factors
        patterns = reasoning_record.get("patterns", [])
        if patterns:
            confidence += len(patterns) * 0.1

        research_insights = reasoning_record.get("research_insights", {})
        if research_insights.get("key_concepts"):
            confidence += 0.1

        if research_insights.get("technologies"):
            confidence += 0.1

        # Cap at 0.95
        return min(confidence, 0.95)

    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get reasoning statistics"""
        if not self.reasoning_history:
            return {"total_reasoning_sessions": 0}

        avg_time = sum(record["reasoning_time"] for record in self.reasoning_history) / len(self.reasoning_history)

        return {
            "total_reasoning_sessions": len(self.reasoning_history),
            "average_reasoning_time": avg_time,
            "most_common_task_types": self._get_common_task_types(),
            "success_rate": sum(1 for r in self.reasoning_history if r.get("success")) / len(self.reasoning_history)
        }

    def _get_common_task_types(self) -> List[Tuple[str, int]]:
        """Get most common task types"""
        task_types = [record.get("task_analysis", {}).get("task_type", "unknown")
                     for record in self.reasoning_history]
        return Counter(task_types).most_common(3)

    async def shutdown(self):
        """Shutdown reasoning engine"""
        try:
            self.logger.info("Shutting down reasoning engine...")

            # Save reasoning history
            await self._save_reasoning_history()

            self.logger.info("Reasoning engine shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down reasoning engine: {e}")

    async def _save_reasoning_history(self):
        """Save reasoning history to file"""
        try:
            history_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'reasoning_history.json')

            os.makedirs(os.path.dirname(history_file), exist_ok=True)

            with open(history_file, 'w') as f:
                json.dump({
                    "history": self.reasoning_history,
                    "stats": self.get_reasoning_stats(),
                    "last_updated": time.time()
                }, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving reasoning history: {e}")