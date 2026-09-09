"""Append-only hash chain for AgentExecutionLog integrity.

Each batch of telemetry records within a chain_id is linked by sequential
SHA-256 hashes.  The first record uses a fixed genesis hash as its
``previous_hash``.  Verification walks the chain and recomputes each hash,
detecting any tampering or gap.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


GENESIS_HASH: str = hashlib.sha256(b"GENESIS").hexdigest()


def compute_record_hash(
    previous_hash: str,
    operation_type: str,
    document_id: int | None,
    node_name: str,
    created_at: str,
    metadata: Any,
) -> str:
    """Deterministic SHA-256 hash linking this record to its predecessor."""
    meta_str = json.dumps(metadata, sort_keys=True, default=str) if metadata else ""
    payload = (
        f"{previous_hash}|{operation_type}|{document_id}|"
        f"{node_name}|{created_at}|{meta_str}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
