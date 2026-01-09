"""
Embedding utilities using sentence-transformers for fast batch processing.

Uses all-MiniLM-L6-v2 model with automatic GPU detection.
Batch processing enables ~30 second embedding of 10k+ texts vs hours with API calls.
"""
import logging
import math
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

# Lazy-loaded model singleton
_model = None
_device = None


def _get_model():
    """Lazy load the sentence-transformers model (singleton)."""
    global _model, _device
    if _model is None:
        from sentence_transformers import SentenceTransformer
        import torch

        # Auto-detect GPU
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model on {_device.upper()}...")

        _model = SentenceTransformer('all-MiniLM-L6-v2', device=_device)
        print(f"Model loaded: all-MiniLM-L6-v2 on {_device.upper()}")

    return _model


def get_embedding(text: str, api_url: str = None, model: str = None, timeout: int = 60) -> Optional[List[float]]:
    """
    Get embedding for single text. Backward compatible signature.

    Note: api_url, model, timeout params are ignored (kept for compatibility).

    Args:
        text: Text to embed
        api_url: Ignored (for backward compatibility)
        model: Ignored (for backward compatibility)
        timeout: Ignored (for backward compatibility)

    Returns:
        List of floats (embedding vector) or None on failure
    """
    if not text or not text.strip():
        return None

    try:
        st_model = _get_model()
        embedding = st_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return None


def get_embeddings_batch(texts: List[str], show_progress: bool = True) -> Dict[str, List[float]]:
    """
    Batch embed all texts at once. Returns dict mapping text -> embedding.

    This is the fast path - processes all texts in one GPU batch call.

    Args:
        texts: List of texts to embed
        show_progress: Whether to show progress bar

    Returns:
        Dict mapping text -> embedding list
    """
    if not texts:
        return {}

    model = _get_model()

    # Filter out empty texts
    valid_texts = [t for t in texts if t and t.strip()]

    if not valid_texts:
        return {}

    print(f"Batch embedding {len(valid_texts)} texts on {_device.upper()}...")

    # Batch encode with progress bar
    embeddings = model.encode(
        valid_texts,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        batch_size=64  # Good balance for RTX 3070
    )

    # Build result dict
    result = {}
    for text, emb in zip(valid_texts, embeddings):
        result[text] = emb.tolist()

    return result


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity in range [-1, 1]. Returns 0.0 on invalid input.
    """
    if not a or not b:
        return 0.0

    if len(a) != len(b):
        logger.warning(f"Vector length mismatch: {len(a)} vs {len(b)}")
        return 0.0

    try:
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0


class EmbeddingCache:
    """
    In-memory cache for embeddings keyed by text.

    Supports batch preloading for fast lookups after initial embedding.
    """

    def __init__(self):
        self._cache: Dict[str, List[float]] = {}
        self._hits = 0
        self._misses = 0

    def preload(self, text_embedding_map: Dict[str, List[float]]):
        """
        Preload cache with batch embeddings.

        Args:
            text_embedding_map: Dict mapping text -> embedding
        """
        self._cache.update(text_embedding_map)
        print(f"Cache preloaded with {len(text_embedding_map)} embeddings")

    def get(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache (None if not found)."""
        if text in self._cache:
            self._hits += 1
            return self._cache[text]
        self._misses += 1
        return None

    def get_or_compute(self, text: str, fn_compute) -> Optional[List[float]]:
        """
        Get cached embedding or compute and cache it.

        Args:
            text: Text to get embedding for
            fn_compute: Function that takes text and returns embedding

        Returns:
            Embedding vector or None if computation failed
        """
        if not text:
            return None

        if text in self._cache:
            self._hits += 1
            return self._cache[text]

        self._misses += 1
        embedding = fn_compute(text)

        if embedding is not None:
            self._cache[text] = embedding

        return embedding

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        """Number of cached embeddings."""
        return len(self._cache)

    @property
    def stats(self) -> dict:
        """Cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "size": self.size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%"
        }
