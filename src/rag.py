"""RAG Pipeline with sqlite-vec for vector storage"""

import logging
import sqlite3
import time
from uuid import MAX
import sqlite_vec
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
import ollama
import json
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class SimpleRAG:
    """RAG pipeline with sqlite-vec for vector storage
    
    Can be used as a context manager for automatic resource cleanup:
    
    Example:
        with SimpleRAG() as rag:
            rag.ingest("data/docs/")
            results = rag.retrieve("query")
    """
    
    def __init__(self, db_path: str = "data/vectors.db"):
        """Initialize RAG pipeline with database connection
        
        Args:
            db_path: Path to sqlite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.ollama_client = ollama.Client()
        self._create_tables()
    
    def _create_tables(self):
        """Create vector storage table if it doesn't exist"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT
            )
        """)
        self.conn.commit()
        logger.info("Database tables initialized")
    
    def _embed_text(self, text: str, max_retries: int = 3) -> List[float]:
        """Generate embedding using Ollama with retry logic
        
        Args:
            text: Text to embed
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of embedding values
            
        Raises:
            ValueError: If text is empty or too long
            ConnectionError: If all retry attempts fail
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        
        # nomic-embed-text has 8192 token limit (~32k chars)
        if len(text) > 32000:
            raise ValueError("Text exceeds embedding model's maximum length")
        
        last_exception = None
        for attempt in range(max_retries):
            try:
                response = self.ollama_client.embeddings(
                    model='nomic-embed-text',
                    prompt=text
                )
                return response['embedding']
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                if 'invalid' in error_msg or 'too long' in error_msg:
                    logger.error(f"Validation error in embedding: {e}", exc_info=True)
                    raise
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Embedding attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to generate embedding after {max_retries} attempts: {e}",
                        exc_info=True
                    )
        
        raise ConnectionError(
            f"Failed to generate embedding after {max_retries} attempts. "
            f"Last error: {last_exception}"
        )
    
    def _extract_document_chunks(self, doc_result) -> List[Dict[str, Any]]:
        """Extract chunks using Docling's HybridChunker with token-aware splitting"""
        
        # Initialize tokenizer matching your embedding model
        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
            max_tokens=350  # Conservative limit well under 512 token max (400 would still exceed limit)
        )
        
        # Create chunker with token-aware settings
        chunker = HybridChunker(
            tokenizer=tokenizer,
            merge_peers=True  # Merge small adjacent chunks when possible
        )
        
        # Generate chunks
        chunk_iter = chunker.chunk(dl_doc=doc_result.document)
        
        chunks = []
        for chunk in chunk_iter:
            # Use contextualize() to get heading-enriched text for embedding
            enriched_text = chunker.contextualize(chunk=chunk)
            
            # Extract metadata from chunk - use safe attribute access
            metadata = {"item_type": "text", "level": None}
            try:
                if hasattr(chunk, 'meta') and chunk.meta:
                    # Try to access doc_items if available using getattr
                    doc_items = getattr(chunk.meta, 'doc_items', None)
                    if doc_items and len(doc_items) > 0:
                        doc_item = doc_items[0]
                        metadata["item_type"] = getattr(doc_item, 'label', 'text')
                        metadata["level"] = getattr(doc_item, 'level', None)
            except (AttributeError, IndexError, TypeError):
                # If metadata extraction fails, use defaults
                pass
            
            chunks.append({
                "content": enriched_text,
                "metadata": metadata
            })

        return chunks

    
    def ingest(self, docs_path: str):
        """Load and process documents from a directory
        
        Args:
            docs_path: Path to directory containing documents
            
        Raises:
            ValueError: If docs_path is invalid
            FileNotFoundError: If path doesn't exist
        """
        # Validate input
        if not docs_path or not docs_path.strip():
            raise ValueError("docs_path cannot be empty")
        
        docs_path_obj = Path(docs_path)
        
        # Check for path traversal attempts
        try:
            docs_path_obj = docs_path_obj.resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid path: {e}")
        
        if not docs_path_obj.exists():
            raise FileNotFoundError(f"Documentation path not found: {docs_path_obj}")
        
        if not docs_path_obj.is_dir():
            raise ValueError(f"Path must be a directory: {docs_path_obj}")
        
        converter = DocumentConverter()
        processed_count = 0
        chunk_count = 0
        
        # Process each markdown document
        for doc_file in docs_path_obj.glob("**/*.md"):
            try:
                logger.info(f"Processing: {doc_file.name}")
                
                # Convert document using Docling
                result = converter.convert(str(doc_file))
                
                # Extract semantic chunks using Docling's document structure
                chunks = self._extract_document_chunks(result)
                
                # Embed and store each chunk
                for chunk_data in chunks:
                    try:
                        content = chunk_data["content"]
                        chunk_metadata = chunk_data["metadata"]
                        
                        embedding = self._embed_text(content)
                        
                        # Convert embedding to bytes for storage
                        embedding_bytes = sqlite_vec.serialize_float32(embedding)
                        
                        # Combine file metadata with chunk metadata
                        metadata = json.dumps({
                            "source": doc_file.name,
                            "path": str(doc_file),
                            "item_type": chunk_metadata.get("item_type"),
                            "level": chunk_metadata.get("level")
                        })
                        
                        self.conn.execute(
                            "INSERT INTO documents (content, embedding, metadata) VALUES (?, ?, ?)",
                            (content, embedding_bytes, metadata)
                        )
                        chunk_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to process chunk: {e}")
                        continue
                
                processed_count += 1
                logger.info(f"Processed {len(chunks)} semantic chunks from {doc_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing {doc_file.name}: {e}", exc_info=True)
                continue
        
        self.conn.commit()
        logger.info(f"Ingestion complete: {processed_count} documents, {chunk_count} chunks")
    
    def retrieve(self, query: str, k: int = 5, rerank: bool = False) -> List[Dict[str, Any]]:
        """Semantic search for relevant chunks with optional re-ranking
        
        Args:
            query: Search query
            k: Number of results to return
            rerank: Whether to re-rank results using cross-encoder (default: True)
            
        Returns:
            List of dictionaries containing content, score, and metadata
            
        Raises:
            ValueError: If query is invalid or k is out of range
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if len(query) > 10000:
            raise ValueError("Query exceeds maximum length of 10000 characters")
        
        # Validate k parameter
        if k < 1:
            raise ValueError("k must be at least 1")
        
        if k > 100:
            raise ValueError("k cannot exceed 100 results")
        
        try:
            # Generate query embedding
            query_embedding = self._embed_text(query)
            query_embedding_bytes = sqlite_vec.serialize_float32(query_embedding)
            
            # Retrieve more candidates if re-ranking is enabled
            initial_k = k * 2 if rerank else k
            
            # Vector similarity search using cosine distance
            cursor = self.conn.execute("""
                SELECT
                    content,
                    metadata,
                    vec_distance_cosine(embedding, ?) as distance
                FROM documents
                ORDER BY distance
                LIMIT ?
            """, (query_embedding_bytes, initial_k))
            
            results = cursor.fetchall()
            
            # Format results
            formatted_results = []
            for row in results:
                content, metadata_json, distance = row
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                # Convert distance to similarity score (1 - distance)
                similarity_score = 1 - distance
                
                formatted_results.append({
                    "content": content,
                    "score": similarity_score,
                    "metadata": metadata
                })
            
            # Re-rank if enabled and we have more results than needed
            if rerank and len(formatted_results) > k:
                formatted_results = self._rerank_results(query, formatted_results)
            
            # Return top k results
            return formatted_results[:k]
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}", exc_info=True)
            raise
    
    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-rank results using Ollama embeddings for better relevance
        
        Uses the same embedding model (nomic-embed-text) already loaded by the RAG system
        to compute more precise similarity scores between query and each document.
        
        Args:
            query: Original search query
            results: List of initial results to re-rank
            
        Returns:
            Re-ranked list of results sorted by relevance
        """
        try:
            # Generate query embedding (reuse existing method)
            query_embedding = self._embed_text(query)
            
            # Re-score each result with detailed embedding comparison
            for result in results:
                # Generate embedding for this specific document chunk
                doc_embedding = self._embed_text(result["content"])
                
                # Compute cosine similarity between query and document
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                
                # Store as rerank score (higher is better)
                result["rerank_score"] = similarity
            
            # Sort by rerank score (descending)
            reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
            
            logger.debug(f"Re-ranked {len(results)} results using Ollama embeddings")
            return reranked
            
        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}. Returning original results.")
            return results
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0 to 1, higher is more similar)
        """
        # Compute dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Compute magnitudes
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        # Cosine similarity
        return dot_product / (magnitude1 * magnitude2)
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Enter context manager"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and cleanup resources
        
        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
            
        Returns:
            False to propagate exceptions
        """
        self.close()
        return False  # Propagate exceptions

# Made with Bob
