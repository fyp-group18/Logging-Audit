# core/config.py — Evaluation pipeline configuration (stripped of Azure/GCS)
import asyncio
import logging
import os
import time
from typing import Callable

from pathlib import Path
from dotenv import load_dotenv

# Load .env.local first (local overrides), then .env (defaults).
_env_dir = Path(__file__).resolve().parent.parent
load_dotenv(_env_dir / ".env.local")
load_dotenv(_env_dir / ".env")

from google import genai
from google.genai import types as genai_types
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmCategory,
    HarmBlockThreshold,
)

logger = logging.getLogger(__name__)

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- Model Name Constants ---
MODEL_PRO = "gemini-2.5-pro"
MODEL_FLASH = "gemini-2.5-flash"

MODEL_EMBEDDING = "gemini-embedding-2"
EMBEDDING_DIM = 3072

# --- Search Configuration ---
SEARCH_VECTOR_CANDIDATES = int(os.getenv("SEARCH_VECTOR_CANDIDATES", "30"))

# --- MMR Diversity ---
SEARCH_MMR_CANDIDATES = int(os.getenv("SEARCH_MMR_CANDIDATES", "100"))
SEARCH_MMR_LAMBDA = float(os.getenv("SEARCH_MMR_LAMBDA", "0.7"))
SEARCH_MMR_ENABLED = os.getenv("SEARCH_MMR_ENABLED", "true").lower() == "true"

# --- Query Rewriting ---
QUERY_REWRITE_ENABLED = os.getenv("QUERY_REWRITE_ENABLED", "true").lower() == "true"

# --- Feedback-Weighted Retrieval ---
FEEDBACK_BOOST_ALPHA = float(os.getenv("FEEDBACK_BOOST_ALPHA", "0.05"))

# --- Retrieval Mode ---
# "collapsed" = search all RAPTOR levels (leaf + summary); "flat" = leaf-only (level 0)
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "collapsed")

# --- Audit Logging Toggle (SYS-01 overhead A/B test) ---
AUDIT_LOGGING_ENABLED = os.getenv("AUDIT_LOGGING_ENABLED", "true").lower() == "true"

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
# gemini-embedding-2 (GA) is available on the global endpoint.
EMBEDDING_LOCATION = "global"

if not PROJECT_ID:
    raise ValueError("CRITICAL: GOOGLE_CLOUD_PROJECT environment variable not found.")

cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if cred_path and os.path.exists(cred_path):
    if not os.path.isabs(cred_path):
        absolute_path = os.path.abspath(cred_path)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = absolute_path
        logger.info(f"Resolved Local Google Credentials to: {absolute_path}")
else:
    logger.info(
        "No local credentials file found. Relying on Google Cloud Run Default Service Account (ADC)."
    )

# 1. Initialize the unified GenAI Client for Vertex AI (used for
#    direct Gemini Pro/Flash calls that run in the project-wide region).
genai_client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

# Dedicated embedding client, pinned to EMBEDDING_LOCATION.
genai_client_embed = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=EMBEDDING_LOCATION,
)


def generate_with_retry(
    model: str,
    contents,
    config: genai_types.GenerateContentConfig | None = None,
    *,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> genai_types.GenerateContentResponse:
    """Wrapper around genai_client.models.generate_content with exponential
    backoff on 429 RESOURCE_EXHAUSTED errors.  The raw google-genai SDK does
    not retry automatically, unlike the LangChain wrapper."""
    for attempt in range(max_retries + 1):
        try:
            return genai_client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            if "429" in str(e) and attempt < max_retries:
                delay = base_delay * (2**attempt)
                logger.warning(
                    f"[generate_with_retry] 429 on {model} (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {delay:.1f}s"
                )
                time.sleep(delay)
            else:
                raise


def stream_with_retry(
    model: str,
    contents,
    config: genai_types.GenerateContentConfig | None = None,
    *,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> str:
    """Streaming variant of generate_with_retry.  Uses
    genai_client.models.generate_content_stream() to keep the SSE
    connection alive during long generations, then returns the
    accumulated text.  Retries on 429 just like the blocking version."""
    for attempt in range(max_retries + 1):
        try:
            accumulated = []
            for chunk in genai_client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            ):
                text = getattr(chunk, "text", None)
                if text:
                    accumulated.append(text)
            return "".join(accumulated)
        except Exception as e:
            if "429" in str(e) and attempt < max_retries:
                delay = base_delay * (2**attempt)
                logger.warning(
                    f"[stream_with_retry] 429 on {model} (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {delay:.1f}s"
                )
                time.sleep(delay)
            else:
                raise


def stream_with_writer(
    model: str,
    contents,
    config: genai_types.GenerateContentConfig | None = None,
    *,
    writer: Callable | None = None,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> str:
    """Streaming variant that forwards raw text chunks to a writer callback
    so the frontend can parse incrementally.  Returns the full accumulated
    text for backend post-processing."""
    for attempt in range(max_retries + 1):
        try:
            accumulated = []
            for chunk in genai_client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            ):
                text = getattr(chunk, "text", None)
                if not text:
                    continue
                accumulated.append(text)
                if writer:
                    writer({"type": "repair_chunk", "chunk": text})
            return "".join(accumulated)
        except Exception as e:
            if "429" in str(e) and attempt < max_retries:
                delay = base_delay * (2**attempt)
                logger.warning(
                    f"[stream_with_writer] 429 on {model} (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {delay:.1f}s"
                )
                time.sleep(delay)
            else:
                raise


# Disable safety filters for industrial terminology
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

# 2. Use ChatGoogleGenerativeAI with your Project and Location
llm_pro_langchain = ChatGoogleGenerativeAI(
    model=MODEL_PRO,
    project=PROJECT_ID,
    location=LOCATION,
    safety_settings=SAFETY_SETTINGS,
    temperature=0.2,
    max_retries=3,
    timeout=300,
)

llm_flash_langchain = ChatGoogleGenerativeAI(
    model=MODEL_FLASH,
    project=PROJECT_ID,
    location=LOCATION,
    safety_settings=SAFETY_SETTINGS,
    temperature=0.1,
    max_retries=3,
    timeout=300,
)
