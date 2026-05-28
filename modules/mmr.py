"""Maximal Marginal Relevance (MMR) diversity reranking.

Greedy selection that balances relevance to the query with diversity
from already-selected candidates. Used by multimodal_semantic_search()
to diversify the vector search candidate pool.
"""

import numpy as np


def mmr_rerank(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_scores: list[float],
    k: int,
    lambda_: float = 0.7,
) -> list[int]:
    """Select k candidates via greedy MMR.

    Args:
        query_embedding: (D,) query vector (unused in scoring but kept for API
            completeness — relevance comes from pre-computed candidate_scores).
        candidate_embeddings: (N, D) candidate vectors.
        candidate_scores: length-N cosine similarities to the query.
        k: number of candidates to select.
        lambda_: tradeoff — 1.0 = pure relevance, 0.0 = pure diversity.

    Returns:
        List of selected indices into the candidate arrays, ordered by
        MMR selection sequence (first = most relevant).
    """
    n = len(candidate_scores)
    if n <= k:
        return list(range(n))

    # L2-normalize for cosine similarity between candidates
    norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    normed = candidate_embeddings / np.maximum(norms, 1e-10)

    scores_arr = np.asarray(candidate_scores, dtype=np.float64)
    remaining_mask = np.ones(n, dtype=bool)
    selected_indices: list[int] = []

    # Seed with the highest-relevance candidate
    best_first = int(np.argmax(scores_arr))
    selected_indices.append(best_first)
    remaining_mask[best_first] = False

    while len(selected_indices) < k and remaining_mask.any():
        # (N, S) similarity of every candidate to each selected
        selected_embs = normed[selected_indices]
        sim_matrix = normed @ selected_embs.T
        max_sim = sim_matrix.max(axis=1)  # (N,)

        # MMR = λ·relevance − (1−λ)·max_similarity_to_selected
        mmr_scores = lambda_ * scores_arr - (1.0 - lambda_) * max_sim
        mmr_scores[~remaining_mask] = -np.inf

        best_idx = int(np.argmax(mmr_scores))
        selected_indices.append(best_idx)
        remaining_mask[best_idx] = False

    return selected_indices
