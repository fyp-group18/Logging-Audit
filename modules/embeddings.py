"""
Embedding layer.

Exposes `embed()` and `embed_batch()` that return 3072-D vectors using
`gemini-embedding-2` (GA) via the dedicated `genai_client_embed`
(global endpoint).

Text + image bytes go through a single `embed_content` call as an
interleaved `contents` list and produce one unified 3072-D vector.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Optional

from google.genai import types

from core.config import MODEL_EMBEDDING, genai_client_embed

_MAX_RETRIES = 15
_BASE_BACKOFF_S = 2
_CONCURRENT_WORKERS = 10

logger = logging.getLogger(__name__)

# Module-level counter for embedding modality outcomes.
_V2_COUNTERS: dict[str, int] = {
    "multimodal_success": 0,
    "text_fallback": 0,
    "text_only_call": 0,
    "failure": 0,
}


def get_v2_embedding_counters() -> dict[str, int]:
    """Return a snapshot of the embedding modality counters."""
    return dict(_V2_COUNTERS)


def reset_v2_embedding_counters() -> None:
    """Zero the embedding modality counters."""
    for k in _V2_COUNTERS:
        _V2_COUNTERS[k] = 0


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is a retryable transient error."""
    msg = str(exc).lower()
    return any(
        tok in msg
        for tok in (
            "429",
            "quota",
            "resource_exhausted",  # rate limit
            "500",
            "503",
            "internal",
            "unavailable",  # server errors
            "timeout",
            "deadline_exceeded",  # timeouts
            "connection",
            "reset",
            "broken pipe",  # network errors
        )
    )


def _call_with_retry(contents, *, label: str):
    """Call embed_content with exponential backoff on rate-limit errors."""
    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES):
        try:
            return genai_client_embed.models.embed_content(
                model=MODEL_EMBEDDING,
                contents=contents,
            )
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if _is_retryable(e) and attempt < _MAX_RETRIES - 1:
                sleep_time = (2**attempt) * _BASE_BACKOFF_S
                logger.warning(
                    f"[{label}] Transient error, retry {attempt + 1}/{_MAX_RETRIES} "
                    f"in {sleep_time}s: {e}"
                )
                time.sleep(sleep_time)
            else:
                raise
    raise last_exc  # type: ignore[misc]


def _flatten_values(values) -> list[float]:
    """Ensure embedding values are a flat list of Python floats.

    The Gemini SDK may return a plain list, a numpy array, a protobuf
    RepeatedScalarContainer, or a nested structure.  ``np.asarray``
    normalises all array-like inputs, then ``.flatten().tolist()``
    guarantees a 1-D Python ``list[float]`` for pgvector.
    """
    import numpy as np

    return np.asarray(values, dtype=np.float64).flatten().tolist()


@lru_cache(maxsize=128)
def _embed_text_cached(text: str) -> Optional[tuple]:
    """Cached text-only embedding. Returns tuple (hashable) for LRU cache."""
    parts = [types.Part.from_text(text=text)]
    try:
        res = _call_with_retry(parts, label="embed-text")
        _V2_COUNTERS["text_only_call"] += 1
        return tuple(_flatten_values(res.embeddings[0].values))
    except Exception as e:  # noqa: BLE001
        logger.error(f"text-only embedding failed: {e}")
        _V2_COUNTERS["failure"] += 1
        return None


def embed(
    text: Optional[str] = None,
    image_bytes_list: Optional[list[bytes]] = None,
) -> Optional[list[float]]:
    """
    Return a single 3072-D embedding for the given interleaved content.

    Text-only embeddings are cached via LRU to avoid duplicate API calls
    for the same query within a request. Multimodal embeddings are not
    cached (image bytes are too large and rarely repeated).

    Returns `None` on unrecoverable API errors.
    """
    if not text and not image_bytes_list:
        logger.warning("embed() called with no text and no images")
        return None

    parts: list = []
    if text:
        parts.append(types.Part.from_text(text=text))
    has_images = False
    if image_bytes_list:
        for img_bytes in image_bytes_list[:4]:
            if not img_bytes:
                continue
            parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
            has_images = True

    if not parts:
        logger.warning(
            "embed() no parts after processing — text=%r, n_images=%d",
            bool(text),
            len(image_bytes_list or []),
        )
        return None

    # Pure text-only call — use LRU cache
    if not has_images:
        result = _embed_text_cached(text)
        return list(result) if result is not None else None

    # Interleaved text + image call (not cached)
    try:
        res = _call_with_retry(parts, label="embed-multimodal")
        _V2_COUNTERS["multimodal_success"] += 1
        _maybe_log_ratio()
        return _flatten_values(res.embeddings[0].values)
    except Exception as e:  # noqa: BLE001
        logger.error(f"multimodal embedding failed: {e}")
        # Fall back to text-only if the interleaved call fails
        if text:
            result = _embed_text_cached(text)
            if result is not None:
                _V2_COUNTERS["text_fallback"] += 1
                _maybe_log_ratio()
                return list(result)
        _V2_COUNTERS["failure"] += 1
        _maybe_log_ratio()
        return None


def embed_batch(texts: list[str]) -> list[Optional[list[float]]]:
    """
    Embed a list of texts concurrently using a thread pool.

    Sends up to ``_CONCURRENT_WORKERS`` requests in parallel. The Vertex AI
    embedding quota (typically 600+ RPM) easily supports this level of
    concurrency, and exponential back-off inside ``_call_with_retry``
    handles transient 429s.
    """
    if not texts:
        return []
    results: list[Optional[list[float]]] = [None] * len(texts)

    def _do(idx: int, text: str) -> tuple[int, Optional[list[float]]]:
        return idx, embed(text=text)

    with ThreadPoolExecutor(max_workers=_CONCURRENT_WORKERS) as pool:
        futures = {pool.submit(_do, i, t): i for i, t in enumerate(texts)}
        for fut in as_completed(futures):
            idx, vec = fut.result()
            results[idx] = vec

    ok = sum(1 for v in results if v is not None)
    logger.info(f"embed_batch complete: {ok}/{len(texts)} succeeded")
    return results


def _maybe_log_ratio() -> None:
    """Log the running multimodal-vs-fallback ratio every 25 attempts."""
    attempts = (
        _V2_COUNTERS["multimodal_success"]
        + _V2_COUNTERS["text_fallback"]
        + _V2_COUNTERS["failure"]
    )
    if attempts == 0 or attempts % 25 != 0:
        return
    mm = _V2_COUNTERS["multimodal_success"]
    fb = _V2_COUNTERS["text_fallback"]
    fail = _V2_COUNTERS["failure"]
    logger.info(
        f"[embed] multimodal={mm} fallback={fb} failure={fail} "
        f"(multimodal_ratio={mm / attempts:.2%}, fallback_ratio={fb / attempts:.2%})"
    )
