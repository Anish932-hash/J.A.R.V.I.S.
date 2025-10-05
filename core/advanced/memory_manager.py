"""
J.A.R.V.I.S. Memory Manager
Persistent memory system with vector database for long-term knowledge retention
"""

import os
import json
import time
import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

# Vector database imports
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Embedding model imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class MemoryManager:
    """
    Advanced memory management system
    Uses vector databases for long-term knowledge retention and retrieval
    """

    def __init__(self, jarvis_instance):
        """
        Initialize memory manager

        Args:
            jarvis_instance: Reference to main JARVIS instance
        """
        self.jarvis = jarvis_instance
        self.logger = logging.getLogger('JARVIS.MemoryManager')

        # Memory storage
        self.short_term_memory: List[Dict[str, Any]] = []
        self.long_term_memory: List[Dict[str, Any]] = []

        # Vector database
        self.vector_db = None
        self.embedding_model = None

        # Memory configuration
        self.config = {
            "short_term_limit": 100,
            "long_term_limit": 10000,
            "similarity_threshold": 0.8,
            "memory_decay_rate": 0.95,
            "auto_cleanup_interval": 3600  # 1 hour
        }

        # Memory statistics
        self.stats = {
            "memories_stored": 0,
            "memories_retrieved": 0,
            "similarity_searches": 0,
            "memory_cleanup_cycles": 0
        }

    async def initialize(self):
        """Initialize memory manager"""
        try:
            self.logger.info("Initializing memory manager...")

            # Initialize vector database
            await self._initialize_vector_database()

            # Load existing memories
            await self._load_existing_memories()

            # Start memory maintenance
            asyncio.create_task(self._memory_maintenance_loop())

            self.logger.info("Memory manager initialized")

        except Exception as e:
            self.logger.error(f"Error initializing memory manager: {e}")
            raise

    async def _initialize_vector_database(self):
        """Initialize vector database and embedding model"""
        try:
            # Initialize embedding model
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.logger.info("Initializing SentenceTransformer embedding model...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and efficient model
                self.logger.info("Embedding model initialized")
            else:
                self.logger.warning("SentenceTransformers not available, using fallback embeddings")

            if CHROMADB_AVAILABLE:
                # Use ChromaDB for vector storage
                persist_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'memory_db')

                self.vector_db = chromadb.PersistentClient(path=persist_dir)

                # Create or get collection
                try:
                    self.memory_collection = self.vector_db.get_or_create_collection(
                        name="jarvis_memories",
                        metadata={"description": "J.A.R.V.I.S. long-term memory"}
                    )
                except Exception as e:
                    self.logger.error(f"Error creating ChromaDB collection: {e}")
                    self.vector_db = None

            elif FAISS_AVAILABLE:
                # Use FAISS as fallback
                self.logger.info("Using FAISS for vector storage")
                # FAISS initialization would go here
            else:
                self.logger.warning("No vector database available, using in-memory storage only")

        except Exception as e:
            self.logger.error(f"Error initializing vector database: {e}")

    async def _load_existing_memories(self):
        """Load existing memories from storage"""
        try:
            memory_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'memories.json')

            if os.path.exists(memory_file):
                with open(memory_file, 'r') as f:
                    memory_data = json.load(f)

                self.long_term_memory = memory_data.get("long_term_memories", [])
                self.short_term_memory = memory_data.get("short_term_memories", [])

                self.logger.info(f"Loaded {len(self.long_term_memory)} long-term memories")

        except Exception as e:
            self.logger.error(f"Error loading existing memories: {e}")

    async def store_memory(self,
                          content: str,
                          memory_type: str = "conversation",
                          metadata: Dict[str, Any] = None,
                          importance: float = 0.5) -> str:
        """
        Store a memory

        Args:
            content: Memory content
            memory_type: Type of memory (conversation, fact, skill, etc.)
            metadata: Additional metadata
            importance: Importance score (0.0 to 1.0)

        Returns:
            Memory ID
        """
        try:
            memory_id = f"mem_{int(time.time())}_{len(self.short_term_memory)}"

            memory = {
                "memory_id": memory_id,
                "content": content,
                "memory_type": memory_type,
                "metadata": metadata or {},
                "importance": importance,
                "created_at": time.time(),
                "access_count": 0,
                "last_accessed": time.time(),
                "embedding": None
            }

            # Store in short-term memory first
            self.short_term_memory.append(memory)

            # Move to long-term if important enough
            if importance > 0.7:
                await self._move_to_long_term(memory)

            # Maintain memory limits
            await self._maintain_memory_limits()

            self.stats["memories_stored"] += 1

            self.logger.debug(f"Stored memory: {memory_id}")

            return memory_id

        except Exception as e:
            self.logger.error(f"Error storing memory: {e}")
            return ""

    async def _move_to_long_term(self, memory: Dict[str, Any]):
        """Move memory to long-term storage"""
        try:
            # Generate embedding for vector search
            embedding = await self._generate_embedding(memory["content"])

            if embedding is not None:
                memory["embedding"] = embedding

                # Store in vector database
                if self.vector_db and self.memory_collection:
                    try:
                        self.memory_collection.add(
                            documents=[memory["content"]],
                            metadatas=[{
                                "memory_id": memory["memory_id"],
                                "memory_type": memory["memory_type"],
                                "importance": memory["importance"],
                                "created_at": memory["created_at"]
                            }],
                            ids=[memory["memory_id"]]
                        )
                    except Exception as e:
                        self.logger.error(f"Error storing in vector DB: {e}")

                # Add to long-term memory
                self.long_term_memory.append(memory)

                # Remove from short-term
                self.short_term_memory = [m for m in self.short_term_memory if m["memory_id"] != memory["memory_id"]]

                self.logger.info(f"Moved memory to long-term: {memory['memory_id']}")

        except Exception as e:
            self.logger.error(f"Error moving memory to long-term: {e}")

    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using SentenceTransformer"""
        try:
            if self.embedding_model:
                # Use SentenceTransformer for high-quality embeddings
                embedding = self.embedding_model.encode(text)
                return embedding.tolist()  # Convert numpy array to list
            else:
                # Fallback to simple hash-based embedding
                self.logger.warning("Using fallback hash-based embedding")
                import hashlib
                hash_obj = hashlib.md5(text.encode())
                hash_bytes = hash_obj.digest()

                # Convert to list of floats (normalized)
                embedding = [float(b) / 255.0 for b in hash_bytes]

                return embedding

        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            return None

    async def retrieve_memories(self,
                               query: str,
                               limit: int = 10,
                               memory_type: str = None,
                               threshold: float = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories

        Args:
            query: Search query
            limit: Maximum number of results
            memory_type: Filter by memory type
            threshold: Similarity threshold

        Returns:
            List of relevant memories
        """
        try:
            threshold = threshold or self.config["similarity_threshold"]
            relevant_memories = []

            # Search in long-term memory using vector similarity
            if self.vector_db and self.memory_collection:
                try:
                    # Generate query embedding
                    query_embedding = await self._generate_embedding(query)

                    if query_embedding:
                        # Search vector database
                        results = self.memory_collection.query(
                            query_embeddings=[query_embedding],
                            n_results=limit,
                            where={"memory_type": memory_type} if memory_type else None
                        )

                        # Process results
                        for i, memory_id in enumerate(results["ids"][0]):
                            if results["distances"][0][i] <= (1 - threshold):  # Convert similarity threshold
                                # Find memory in long-term storage
                                memory = next((m for m in self.long_term_memory if m["memory_id"] == memory_id), None)
                                if memory:
                                    memory["similarity"] = 1 - results["distances"][0][i]
                                    memory["access_count"] += 1
                                    memory["last_accessed"] = time.time()
                                    relevant_memories.append(memory)

                except Exception as e:
                    self.logger.error(f"Error searching vector database: {e}")

            # Fallback to text-based search in short-term memory
            if len(relevant_memories) < limit:
                short_term_results = self._search_short_term_memory(query, limit - len(relevant_memories), memory_type)
                relevant_memories.extend(short_term_results)

            self.stats["memories_retrieved"] += len(relevant_memories)

            return relevant_memories

        except Exception as e:
            self.logger.error(f"Error retrieving memories: {e}")
            return []

    def _search_short_term_memory(self,
                                 query: str,
                                 limit: int,
                                 memory_type: str = None) -> List[Dict[str, Any]]:
        """Search short-term memory using text matching"""
        results = []

        query_terms = set(query.lower().split())

        for memory in self.short_term_memory:
            if memory_type and memory["memory_type"] != memory_type:
                continue

            content = memory["content"].lower()
            score = sum(1 for term in query_terms if term in content)

            if score > 0:
                memory["similarity"] = score / len(query_terms)
                memory["access_count"] += 1
                memory["last_accessed"] = time.time()
                results.append(memory)

        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return results[:limit]

    async def _maintain_memory_limits(self):
        """Maintain memory storage limits"""
        try:
            # Short-term memory cleanup
            if len(self.short_term_memory) > self.config["short_term_limit"]:
                # Remove oldest, least important memories
                self.short_term_memory.sort(key=lambda x: (x["importance"], x["last_accessed"]))
                self.short_term_memory = self.short_term_memory[:self.config["short_term_limit"]]

            # Long-term memory cleanup (less frequent)
            if len(self.long_term_memory) > self.config["long_term_limit"]:
                # Apply decay and remove least important
                for memory in self.long_term_memory:
                    # Apply decay factor
                    time_since_creation = time.time() - memory["created_at"]
                    decay_factor = self.config["memory_decay_rate"] ** (time_since_creation / 86400)  # Daily decay
                    memory["importance"] *= decay_factor

                # Sort by decayed importance
                self.long_term_memory.sort(key=lambda x: x["importance"])
                self.long_term_memory = self.long_term_memory[:self.config["long_term_limit"]]

        except Exception as e:
            self.logger.error(f"Error maintaining memory limits: {e}")

    async def _memory_maintenance_loop(self):
        """Memory maintenance background loop"""
        while True:
            try:
                await asyncio.sleep(self.config["auto_cleanup_interval"])

                # Run maintenance
                await self._maintain_memory_limits()

                # Save memories to disk
                await self._save_memories_to_disk()

                self.stats["memory_cleanup_cycles"] += 1

            except Exception as e:
                self.logger.error(f"Error in memory maintenance: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry

    async def _save_memories_to_disk(self):
        """Save memories to persistent storage"""
        try:
            memory_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'memories.json')

            memory_data = {
                "short_term_memories": self.short_term_memory,
                "long_term_memories": self.long_term_memory,
                "last_saved": time.time(),
                "stats": self.stats
            }

            os.makedirs(os.path.dirname(memory_file), exist_ok=True)
            with open(memory_file, 'w') as f:
                json.dump(memory_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving memories: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return {
            **self.stats,
            "short_term_memories": len(self.short_term_memory),
            "long_term_memories": len(self.long_term_memory),
            "vector_db_available": self.vector_db is not None,
            "total_memories": len(self.short_term_memory) + len(self.long_term_memory)
        }

    def clear_memory(self, memory_type: str = None):
        """Clear memories"""
        try:
            if memory_type == "short_term" or memory_type is None:
                self.short_term_memory.clear()

            if memory_type == "long_term" or memory_type is None:
                self.long_term_memory.clear()

                # Clear vector database
                if self.vector_db and self.memory_collection:
                    try:
                        self.vector_db.delete_collection("jarvis_memories")
                        self.memory_collection = self.vector_db.create_collection(
                            name="jarvis_memories",
                            metadata={"description": "J.A.R.V.I.S. long-term memory"}
                        )
                    except Exception as e:
                        self.logger.error(f"Error clearing vector database: {e}")

            self.logger.info(f"Cleared {memory_type or 'all'} memories")

        except Exception as e:
            self.logger.error(f"Error clearing memory: {e}")

    async def search_similar_memories(self,
                                    query: str,
                                    limit: int = 5,
                                    threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar memories using vector similarity"""
        return await self.retrieve_memories(query, limit, threshold=threshold)

    async def get_memory_context(self, current_query: str) -> str:
        """Get relevant context from memory for current query"""
        try:
            # Retrieve relevant memories
            relevant_memories = await self.retrieve_memories(current_query, limit=3)

            if not relevant_memories:
                return ""

            # Build context string
            context_parts = []
            for memory in relevant_memories:
                if memory.get("similarity", 0) > 0.6:  # Only include highly relevant memories
                    context_parts.append(f"Previous context: {memory['content']}")

            return "\n".join(context_parts) if context_parts else ""

        except Exception as e:
            self.logger.error(f"Error getting memory context: {e}")
            return ""

    async def update_memory_importance(self, memory_id: str, new_importance: float):
        """Update importance of a memory"""
        try:
            # Find memory in either storage
            for memory in self.short_term_memory + self.long_term_memory:
                if memory["memory_id"] == memory_id:
                    memory["importance"] = new_importance

                    # Move between storages if importance threshold crossed
                    if new_importance > 0.7 and memory in self.short_term_memory:
                        await self._move_to_long_term(memory)
                    elif new_importance <= 0.7 and memory in self.long_term_memory:
                        # Move back to short-term
                        self.short_term_memory.append(memory)
                        self.long_term_memory.remove(memory)

                    break

        except Exception as e:
            self.logger.error(f"Error updating memory importance: {e}")

    async def forget_memories(self, criteria: Dict[str, Any]):
        """Forget memories based on criteria"""
        try:
            memories_to_remove = []

            # Find memories matching criteria
            for memory in self.short_term_memory + self.long_term_memory:
                match = True

                for key, value in criteria.items():
                    if key == "older_than_days":
                        age_days = (time.time() - memory["created_at"]) / 86400
                        if age_days <= value:
                            match = False
                            break
                    elif key == "memory_type":
                        if memory["memory_type"] != value:
                            match = False
                            break
                    elif key == "importance_below":
                        if memory["importance"] >= value:
                            match = False
                            break

                if match:
                    memories_to_remove.append(memory)

            # Remove memories
            for memory in memories_to_remove:
                if memory in self.short_term_memory:
                    self.short_term_memory.remove(memory)
                if memory in self.long_term_memory:
                    self.long_term_memory.remove(memory)

                    # Remove from vector database
                    if self.vector_db and self.memory_collection:
                        try:
                            self.memory_collection.delete(ids=[memory["memory_id"]])
                        except:
                            pass

            self.logger.info(f"Forgot {len(memories_to_remove)} memories based on criteria")

        except Exception as e:
            self.logger.error(f"Error forgetting memories: {e}")

    async def export_memories(self, file_path: str) -> bool:
        """Export memories to file"""
        try:
            export_data = {
                "export_timestamp": time.time(),
                "short_term_memories": self.short_term_memory,
                "long_term_memories": self.long_term_memory,
                "stats": self.stats
            }

            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)

            return True

        except Exception as e:
            self.logger.error(f"Error exporting memories: {e}")
            return False

    async def import_memories(self, file_path: str) -> bool:
        """Import memories from file"""
        try:
            with open(file_path, 'r') as f:
                import_data = json.load(f)

            # Merge imported memories
            imported_short = import_data.get("short_term_memories", [])
            imported_long = import_data.get("long_term_memories", [])

            self.short_term_memory.extend(imported_short)
            self.long_term_memory.extend(imported_long)

            # Maintain limits
            await self._maintain_memory_limits()

            self.logger.info(f"Imported {len(imported_short)} short-term and {len(imported_long)} long-term memories")

            return True

        except Exception as e:
            self.logger.error(f"Error importing memories: {e}")
            return False

    async def shutdown(self):
        """Shutdown memory manager"""
        try:
            self.logger.info("Shutting down memory manager...")

            # Save memories to disk
            await self._save_memories_to_disk()

            # Close vector database
            if self.vector_db:
                self.vector_db = None

            self.logger.info("Memory manager shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down memory manager: {e}")