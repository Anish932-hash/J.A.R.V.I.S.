"""
J.A.R.V.I.S. Info Collector
Advanced data analysis and processing for research results
"""

import os
import time
import json
import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
import statistics
from datetime import datetime, timedelta
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    nltk = None
    NLTK_AVAILABLE = False
import math


class TextAnalyzer:
    """Advanced text analysis for research data"""

    def __init__(self):
        if NLTK_AVAILABLE:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        else:
            self.lemmatizer = None
            self.stop_words = set()

        # Programming-specific terms
        self.programming_terms = {
            'python', 'javascript', 'java', 'cpp', 'c++', 'csharp', 'c#', 'php', 'ruby', 'go',
            'rust', 'swift', 'kotlin', 'scala', 'typescript', 'html', 'css', 'sql',
            'function', 'class', 'method', 'variable', 'object', 'array', 'list', 'dict',
            'string', 'integer', 'float', 'boolean', 'api', 'database', 'server', 'client',
            'framework', 'library', 'module', 'package', 'algorithm', 'data', 'structure',
            'performance', 'optimization', 'security', 'testing', 'debugging', 'deployment'
        }

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Comprehensive text analysis"""
        if not text:
            return {}

        # Basic metrics
        sentences = sent_tokenize(text) if nltk else text.split('.')
        words = word_tokenize(text.lower()) if nltk else text.lower().split()

        # Filter out stop words and punctuation
        filtered_words = [word for word in words if word.isalnum() and word not in self.stop_words]

        # Lemmatize words
        lemmas = [self.lemmatizer.lemmatize(word) for word in filtered_words] if nltk else filtered_words

        # Extract programming concepts
        programming_concepts = [word for word in lemmas if word in self.programming_terms]

        # Calculate readability metrics
        readability = self._calculate_readability(text, sentences, words)

        # Extract key phrases (simplified)
        key_phrases = self._extract_key_phrases(text)

        # Sentiment analysis (basic)
        sentiment = self._analyze_sentiment(text)

        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "unique_words": len(set(filtered_words)),
            "programming_concepts": list(set(programming_concepts)),
            "key_phrases": key_phrases,
            "readability_score": readability,
            "sentiment": sentiment,
            "complexity_score": self._calculate_complexity(filtered_words, programming_concepts)
        }

    def _calculate_readability(self, text: str, sentences: List[str], words: List[str]) -> float:
        """Calculate readability score (simplified)"""
        if not sentences or not words:
            return 0.0

        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)

        # Simple readability formula
        readability = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_word_length

        # Normalize to 0-100 scale
        return max(0, min(100, readability))

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        # Simple noun phrase extraction
        phrases = []

        # Look for patterns like "adjective + noun" or "noun + noun"
        words = re.findall(r'\b\w+\b', text.lower())

        for i in range(len(words) - 1):
            if (words[i].isalpha() and len(words[i]) > 2 and
                words[i + 1].isalpha() and len(words[i + 1]) > 2):
                phrase = f"{words[i]} {words[i + 1]}"
                if phrase not in phrases and len(phrases) < 10:
                    phrases.append(phrase)

        return phrases

    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Basic sentiment analysis"""
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'perfect'}
        negative_words = {'bad', 'terrible', 'awful', 'worst', 'horrible', 'poor', 'fail'}

        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        total_opinion_words = positive_count + negative_count

        if total_opinion_words == 0:
            sentiment_score = 0.5
        else:
            sentiment_score = positive_count / total_opinion_words

        return {
            "score": sentiment_score,
            "positive_words": positive_count,
            "negative_words": negative_count,
            "label": "positive" if sentiment_score > 0.6 else "negative" if sentiment_score < 0.4 else "neutral"
        }

    def _calculate_complexity(self, words: List[str], programming_concepts: List[str]) -> float:
        """Calculate text complexity score"""
        if not words:
            return 0.0

        # Factors contributing to complexity
        avg_word_length = sum(len(word) for word in words) / len(words)
        unique_ratio = len(set(words)) / len(words)
        programming_ratio = len(programming_concepts) / len(words) if words else 0

        # Weighted complexity score
        complexity = (
            avg_word_length * 0.3 +
            unique_ratio * 0.4 +
            programming_ratio * 0.3
        )

        return min(1.0, complexity)


class DataAnalyzer:
    """Advanced data analysis for research results"""

    def __init__(self):
        self.text_analyzer = TextAnalyzer()

    def analyze_dataset(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze entire dataset"""
        if not data:
            return {"error": "No data to analyze"}

        # Aggregate analysis
        all_texts = []
        sources = Counter()
        timestamps = []

        for item in data:
            # Extract text content
            text_content = self._extract_text_content(item)
            if text_content:
                all_texts.append(text_content)

            # Track sources
            source = item.get('source', 'unknown')
            sources[source] += 1

            # Track timestamps
            timestamp = item.get('timestamp')
            if timestamp:
                timestamps.append(timestamp)

        # Analyze all text combined
        combined_text = ' '.join(all_texts)
        text_analysis = self.text_analyzer.analyze_text(combined_text)

        # Temporal analysis
        temporal_stats = self._analyze_temporal_distribution(timestamps)

        # Source analysis
        source_analysis = self._analyze_sources(sources)

        # Content clustering (simplified)
        content_clusters = self._cluster_content(data)

        return {
            "total_items": len(data),
            "text_analysis": text_analysis,
            "temporal_analysis": temporal_stats,
            "source_analysis": source_analysis,
            "content_clusters": content_clusters,
            "data_quality_score": self._calculate_data_quality(data)
        }

    def _extract_text_content(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract text content from data item"""
        text_fields = ['content', 'snippet', 'description', 'text', 'body']

        for field in text_fields:
            if field in item and item[field]:
                return str(item[field])

        return None

    def _analyze_temporal_distribution(self, timestamps: List[float]) -> Dict[str, Any]:
        """Analyze temporal distribution of data"""
        if not timestamps:
            return {"error": "No timestamps available"}

        # Convert to datetime objects
        datetimes = [datetime.fromtimestamp(ts) for ts in timestamps]

        # Calculate time span
        if len(datetimes) > 1:
            time_span = max(datetimes) - min(datetimes)
            avg_interval = time_span / (len(datetimes) - 1)
        else:
            time_span = timedelta(0)
            avg_interval = timedelta(0)

        # Group by day
        day_counts = Counter(dt.date() for dt in datetimes)

        return {
            "total_timespan_days": time_span.days,
            "average_interval_hours": avg_interval.total_seconds() / 3600,
            "unique_days": len(day_counts),
            "most_active_day": day_counts.most_common(1)[0][0].isoformat() if day_counts else None,
            "daily_distribution": dict(day_counts)
        }

    def _analyze_sources(self, sources: Counter) -> Dict[str, Any]:
        """Analyze data sources"""
        total_sources = sum(sources.values())

        return {
            "total_sources": len(sources),
            "source_distribution": dict(sources),
            "most_common_source": sources.most_common(1)[0][0] if sources else None,
            "source_diversity_score": len(sources) / total_sources if total_sources > 0 else 0
        }

    def _cluster_content(self, data: List[Dict[str, Any]], num_clusters: int = 3) -> List[Dict[str, Any]]:
        """Simple content clustering based on keywords"""
        clusters = []

        # Extract keywords from each item
        item_keywords = []
        for item in data:
            text = self._extract_text_content(item) or ""
            keywords = set(re.findall(r'\b\w+\b', text.lower())[:10])  # Top 10 words
            item_keywords.append(keywords)

        # Simple clustering by keyword overlap
        for i in range(min(num_clusters, len(data))):
            cluster_items = []
            cluster_keywords = set()

            # Start with item i
            if i < len(data):
                cluster_items.append(data[i])
                cluster_keywords.update(item_keywords[i])

                # Add similar items
                for j, item in enumerate(data):
                    if j != i and item not in cluster_items:
                        overlap = len(item_keywords[j] & cluster_keywords)
                        if overlap > 2:  # Similarity threshold
                            cluster_items.append(item)
                            cluster_keywords.update(item_keywords[j])

            if cluster_items:
                clusters.append({
                    "size": len(cluster_items),
                    "keywords": list(cluster_keywords)[:5],  # Top 5 keywords
                    "representative_item": cluster_items[0]
                })

        return clusters

    def _calculate_data_quality(self, data: List[Dict[str, Any]]) -> float:
        """Calculate overall data quality score"""
        if not data:
            return 0.0

        quality_score = 0.0
        total_checks = 0

        for item in data:
            # Check for required fields
            has_text = bool(self._extract_text_content(item))
            has_source = bool(item.get('source'))
            has_timestamp = bool(item.get('timestamp'))

            item_quality = (has_text + has_source + has_timestamp) / 3
            quality_score += item_quality
            total_checks += 1

        return quality_score / total_checks if total_checks > 0 else 0.0


class InfoCollector:
    """Advanced data analysis and processing for research results"""

    def __init__(self, development_engine):
        self.engine = development_engine
        self.jarvis = development_engine.jarvis
        self.logger = logging.getLogger('JARVIS.InfoCollector')

        # Analysis components
        self.data_analyzer = DataAnalyzer()
        self.analysis_history = []

        # Analysis cache
        self.analysis_cache = {}
        self.cache_ttl = 3600  # 1 hour

    async def initialize(self):
        """Initialize info collector"""
        try:
            self.logger.info("Initializing info collector...")

            # Try to download NLTK data if available
            try:
                import nltk
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                self.logger.info("✓ NLTK data downloaded")
            except Exception as e:
                self.logger.warning(f"NLTK data download failed: {e}")

            self.logger.info("Info collector initialized")

        except Exception as e:
            self.logger.error(f"Error initializing info collector: {e}")
            raise

    async def analyze_data(self, data: List[Dict[str, Any]], task_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive data analysis for research results"""
        try:
            self.logger.info(f"Analyzing {len(data)} data items")

            analysis_start = time.time()

            # Check cache
            cache_key = self._generate_cache_key(data, task_requirements)
            if cache_key in self.analysis_cache:
                cached_time, cached_result = self.analysis_cache[cache_key]
                if time.time() - cached_time < self.cache_ttl:
                    self.logger.info("Using cached analysis results")
                    return cached_result

            # Perform comprehensive analysis
            dataset_analysis = self.data_analyzer.analyze_dataset(data)

            # Task-specific analysis
            task_specific_analysis = self._analyze_task_requirements(data, task_requirements)

            # Generate insights and recommendations
            insights = self._generate_insights(dataset_analysis, task_specific_analysis)

            # Quality assessment
            quality_assessment = self._assess_data_quality(data, dataset_analysis)

            # Synthesis
            synthesis = self._synthesize_findings(dataset_analysis, task_specific_analysis, insights)

            analysis_time = time.time() - analysis_start

            result = {
                "success": True,
                "dataset_analysis": dataset_analysis,
                "task_specific_analysis": task_specific_analysis,
                "insights": insights,
                "quality_assessment": quality_assessment,
                "synthesis": synthesis,
                "analysis_time": analysis_time,
                "data_items_processed": len(data)
            }

            # Cache result
            self.analysis_cache[cache_key] = (time.time(), result)

            # Record analysis
            self.analysis_history.append({
                "timestamp": time.time(),
                "data_items": len(data),
                "analysis_time": analysis_time,
                "insights_generated": len(insights.get("key_insights", [])),
                "quality_score": quality_assessment.get("overall_score", 0)
            })

            self.logger.info(f"Data analysis completed in {analysis_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error in data analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "basic_stats": {
                    "total_items": len(data),
                    "has_content": sum(1 for item in data if self.data_analyzer._extract_text_content(item))
                }
            }

    def _generate_cache_key(self, data: List[Dict[str, Any]], task_requirements: Dict[str, Any]) -> str:
        """Generate cache key for analysis"""
        import hashlib

        # Create a hash based on data content and requirements
        data_str = json.dumps(data, sort_keys=True, default=str)
        req_str = json.dumps(task_requirements, sort_keys=True, default=str)

        combined = f"{data_str}:{req_str}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _analyze_task_requirements(self, data: List[Dict[str, Any]], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data in context of task requirements"""
        analysis = {
            "requirement_coverage": {},
            "missing_information": [],
            "relevant_data_items": 0,
            "requirement_matches": {}
        }

        if not requirements:
            return analysis

        # Analyze each requirement
        for req_key, req_value in requirements.items():
            coverage = self._analyze_requirement_coverage(data, req_key, req_value)
            analysis["requirement_coverage"][req_key] = coverage

            if coverage["coverage_score"] < 0.5:
                analysis["missing_information"].append({
                    "requirement": req_key,
                    "gap": coverage["missing_aspects"]
                })

        # Count relevant data items
        analysis["relevant_data_items"] = sum(
            1 for item in data
            if self._is_relevant_to_requirements(item, requirements)
        )

        return analysis

    def _analyze_requirement_coverage(self, data: List[Dict[str, Any]], req_key: str, req_value: Any) -> Dict[str, Any]:
        """Analyze how well data covers a specific requirement"""
        coverage = {
            "coverage_score": 0.0,
            "matching_items": 0,
            "total_items": len(data),
            "missing_aspects": []
        }

        if not data:
            return coverage

        matching_items = 0

        for item in data:
            text = self.data_analyzer._extract_text_content(item) or ""

            # Check if requirement is mentioned
            req_terms = str(req_value).lower().split()
            matches = sum(1 for term in req_terms if term in text.lower())

            if matches > 0:
                matching_items += 1

        coverage["matching_items"] = matching_items
        coverage["coverage_score"] = matching_items / len(data) if data else 0

        # Identify missing aspects (simplified)
        if coverage["coverage_score"] < 0.3:
            coverage["missing_aspects"] = [f"More information about {req_key}"]

        return coverage

    def _is_relevant_to_requirements(self, item: Dict[str, Any], requirements: Dict[str, Any]) -> bool:
        """Check if data item is relevant to task requirements"""
        text = self.data_analyzer._extract_text_content(item) or ""
        text_lower = text.lower()

        # Check if any requirement terms appear in the text
        for req_value in requirements.values():
            req_terms = str(req_value).lower().split()
            if any(term in text_lower for term in req_terms):
                return True

        return False

    def _generate_insights(self, dataset_analysis: Dict[str, Any], task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from analysis results"""
        insights = {
            "key_insights": [],
            "recommendations": [],
            "trends": [],
            "gaps": []
        }

        # Extract key insights from text analysis
        text_analysis = dataset_analysis.get("text_analysis", {})

        if text_analysis.get("programming_concepts"):
            concepts = text_analysis["programming_concepts"][:5]  # Top 5
            insights["key_insights"].append(f"Key programming concepts: {', '.join(concepts)}")

        # Quality insights
        quality = dataset_analysis.get("data_quality_score", 0)
        if quality < 0.5:
            insights["gaps"].append("Data quality is low - consider gathering more reliable sources")
        elif quality > 0.8:
            insights["key_insights"].append("High-quality data sources identified")

        # Temporal insights
        temporal = dataset_analysis.get("temporal_analysis", {})
        if temporal.get("total_timespan_days", 0) > 30:
            insights["trends"].append("Data spans multiple months - consider temporal analysis")

        # Source insights
        source_analysis = dataset_analysis.get("source_analysis", {})
        if source_analysis.get("source_diversity_score", 0) < 0.3:
            insights["gaps"].append("Limited source diversity - consider expanding search")

        # Task-specific insights
        req_coverage = task_analysis.get("requirement_coverage", {})
        for req, coverage in req_coverage.items():
            if coverage.get("coverage_score", 0) < 0.5:
                insights["gaps"].append(f"Insufficient information about {req}")

        # Generate recommendations
        if insights["gaps"]:
            insights["recommendations"].append("Conduct additional research to fill identified gaps")

        if text_analysis.get("sentiment", {}).get("label") == "negative":
            insights["recommendations"].append("Review negative feedback and address concerns")

        return insights

    def _assess_data_quality(self, data: List[Dict[str, Any]], dataset_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall data quality"""
        assessment = {
            "overall_score": 0.0,
            "dimensions": {
                "completeness": 0.0,
                "accuracy": 0.0,
                "consistency": 0.0,
                "timeliness": 0.0,
                "relevance": 0.0
            },
            "issues": []
        }

        # Completeness
        text_items = sum(1 for item in data if self.data_analyzer._extract_text_content(item))
        assessment["dimensions"]["completeness"] = text_items / len(data) if data else 0

        # Accuracy (estimated)
        assessment["dimensions"]["accuracy"] = dataset_analysis.get("data_quality_score", 0.5)

        # Consistency (simplified)
        sources = [item.get('source') for item in data]
        unique_sources = len(set(sources))
        assessment["dimensions"]["consistency"] = unique_sources / len(data) if data else 0

        # Timeliness
        temporal = dataset_analysis.get("temporal_analysis", {})
        timespan_days = temporal.get("total_timespan_days", 0)
        assessment["dimensions"]["timeliness"] = min(1.0, timespan_days / 365)  # Prefer recent data

        # Relevance (estimated from task analysis)
        relevant_items = sum(1 for item in data if any(self.data_analyzer._extract_text_content(item) for _ in [item]))
        assessment["dimensions"]["relevance"] = relevant_items / len(data) if data else 0

        # Overall score
        assessment["overall_score"] = statistics.mean(assessment["dimensions"].values())

        # Identify issues
        for dimension, score in assessment["dimensions"].items():
            if score < 0.5:
                assessment["issues"].append(f"Low {dimension} score: {score:.2f}")

        return assessment

    def _synthesize_findings(self, dataset_analysis: Dict[str, Any],
                           task_analysis: Dict[str, Any],
                           insights: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize all findings into coherent summary"""
        synthesis = {
            "summary": "",
            "confidence_level": "medium",
            "action_items": [],
            "next_steps": []
        }

        # Generate summary
        total_items = dataset_analysis.get("total_items", 0)
        quality_score = dataset_analysis.get("data_quality_score", 0)
        relevant_items = task_analysis.get("relevant_data_items", 0)

        summary_parts = [
            f"Analyzed {total_items} data items",
            f"Data quality: {'high' if quality_score > 0.8 else 'medium' if quality_score > 0.5 else 'low'}",
            f"Task relevance: {relevant_items}/{total_items} items"
        ]

        if insights.get("key_insights"):
            summary_parts.append(f"Key insights: {len(insights['key_insights'])} identified")

        synthesis["summary"] = ". ".join(summary_parts)

        # Determine confidence
        if quality_score > 0.8 and relevant_items > total_items * 0.7:
            synthesis["confidence_level"] = "high"
        elif quality_score < 0.4 or relevant_items < total_items * 0.3:
            synthesis["confidence_level"] = "low"

        # Action items
        if insights.get("gaps"):
            synthesis["action_items"].extend([f"Address gap: {gap}" for gap in insights["gaps"]])

        if insights.get("recommendations"):
            synthesis["next_steps"].extend(insights["recommendations"])

        return synthesis

    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        if not self.analysis_history:
            return {"total_analyses": 0}

        avg_time = statistics.mean(record["analysis_time"] for record in self.analysis_history)
        avg_quality = statistics.mean(record["quality_score"] for record in self.analysis_history)

        return {
            "total_analyses": len(self.analysis_history),
            "average_analysis_time": avg_time,
            "average_data_quality": avg_quality,
            "cache_size": len(self.analysis_cache),
            "cache_hit_rate": 0  # Would need to track cache hits
        }

    async def shutdown(self):
        """Shutdown info collector"""
        try:
            self.logger.info("Shutting down info collector...")

            # Save analysis history
            await self._save_analysis_history()

            # Clear cache
            self.analysis_cache.clear()

            self.logger.info("Info collector shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down info collector: {e}")

    async def _save_analysis_history(self):
        """Save analysis history to file"""
        try:
            history_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'analysis_history.json')

            os.makedirs(os.path.dirname(history_file), exist_ok=True)

            with open(history_file, 'w') as f:
                json.dump({
                    "history": self.analysis_history,
                    "stats": self.get_analysis_stats(),
                    "last_updated": time.time()
                }, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving analysis history: {e}")