import json
import os
import base64
import logging
import time

from google.genai import types
from pydantic import BaseModel, Field

from core.config import MODEL_FLASH, generate_with_retry

logger = logging.getLogger(__name__)


class ChunkScore(BaseModel):
    chunk_id: str
    score: int = Field(ge=0, le=10, description="Relevance score 0-10")


class RerankerResponse(BaseModel):
    scores: list[ChunkScore]


def llm_rerank_chunks(
    user_query: str,
    user_image_base64: str,
    chunks: list,
    min_score: int = 3,
    top_k: int = 5,
) -> tuple[list, dict, dict]:
    """Rerank chunks via LLM and return (survivors, scores_map, meta).

    Returns:
        reranked_chunks: filtered/sorted chunk list
        scores_map: {chunk_id: int(0-10)} for ALL input chunks
        reranker_meta: timing, token counts, filter stats
    """
    candidates_in = len(chunks)
    if not chunks:
        return [], {}, {"candidates_in": 0, "candidates_out": 0}

    gcs_bucket = os.getenv("GCS_BUCKET_NAME", "").replace('"', "").replace("'", "")

    # Build multimodal content: interleave chunk text with actual images
    # for image-heavy chunks so the reranker can evaluate diagram content
    # (not just generic captions like "Shows assembly step").
    contents: list = []

    if user_image_base64:
        try:
            img_data = base64.b64decode(user_image_base64)
            mime = "image/png" if img_data[:4] == b"\x89PNG" else "image/jpeg"
            contents.append(types.Part.from_bytes(data=img_data, mime_type=mime))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[Reranker] failed to decode user image: {e}")

    prompt_header = (
        f"You are a relevance reranker for industrial equipment technical manuals.\n"
        f"Score each CHUNK 0-10 based on how well it answers the user query.\n\n"
        f"SCORING GUIDE:\n"
        f"  9-10: Chunk directly contains the fault code, troubleshooting step, or exact specification asked about\n"
        f"  7-8:  Chunk covers the correct system/component and contains related diagnostic info\n"
        f"  4-6:  Chunk is from the correct manual section but only partially relevant\n"
        f"  1-3:  Chunk is from the same manual but wrong section/topic\n"
        f"  0:    Completely irrelevant\n\n"
        f'User Query: "{user_query}"\n\nChunks:\n'
    )
    contents.append(prompt_header)

    for c in chunks:
        caption_block = ""
        chunk_images = c.get("images", []) or []
        for img in chunk_images:
            caption = img.get("caption") if isinstance(img, dict) else ""
            if caption:
                caption_block += f"[diagram: {caption}]\n"

        contents.append(
            f"\n--- CHUNK ID: {c['id']} (pages {c.get('pages', '')}) ---\n"
            f"{caption_block}{c['text'][:2000]}\n"
        )

        # For image-heavy chunks with little text, include actual images
        # via GCS URI so Gemini Flash can see diagram content.
        text_len = len(c.get("text", ""))
        if chunk_images and text_len < 200 and gcs_bucket:
            for img in chunk_images[:2]:
                img_path = img.get("path", "") if isinstance(img, dict) else ""
                if img_path and not img_path.startswith("http"):
                    mime = (
                        "image/jpeg"
                        if img_path.lower().endswith((".jpg", ".jpeg"))
                        else "image/png"
                    )
                    try:
                        contents.append(
                            types.Part.from_uri(
                                file_uri=f"gs://{gcs_bucket}/{img_path}",
                                mime_type=mime,
                            )
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"[Reranker] GCS image load failed: {e}")

    try:
        t0 = time.time()
        res = generate_with_retry(
            model=MODEL_FLASH,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RerankerResponse,
                temperature=0.0,
            ),
        )
        latency_ms = int((time.time() - t0) * 1000)

        scores_data = json.loads(res.text).get("scores", [])
        scores_map = {s["chunk_id"]: s["score"] for s in scores_data}
        for c in chunks:
            c["reranker_score"] = scores_map.get(c["id"], 0)
        reranked = sorted(
            chunks, key=lambda x: x.get("reranker_score", 0), reverse=True
        )

        # Extract token usage from response metadata
        usage = getattr(res, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        output_tokens = (
            getattr(usage, "candidates_token_count", None) if usage else None
        )

        min_score_applied = False
        fallback_to_floor = False
        if min_score > 0:
            filtered = [c for c in reranked if c.get("reranker_score", 0) >= min_score]
            min_score_applied = len(filtered) != len(reranked)
            # Floor of 2, ceiling of top_k
            if len(filtered) < 2:
                filtered = reranked[:2]
                fallback_to_floor = True
            result_chunks = filtered[:top_k]
        else:
            result_chunks = reranked[:top_k]

        meta = {
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "fallback_triggered": False,
            "min_score_applied": min_score_applied,
            "fallback_to_floor": fallback_to_floor,
            "candidates_in": candidates_in,
            "candidates_out": len(result_chunks),
        }
        return result_chunks, scores_map, meta

    except Exception as e:
        logger.exception("[Reranker] LLM rerank failed; falling back to top %d", top_k)
        result_chunks = chunks[:top_k]
        meta = {
            "latency_ms": None,
            "input_tokens": None,
            "output_tokens": None,
            "fallback_triggered": True,
            "min_score_applied": False,
            "candidates_in": candidates_in,
            "candidates_out": len(result_chunks),
            "error": str(e),
        }
        return result_chunks, {}, meta
