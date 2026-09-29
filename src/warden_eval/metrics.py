from typing import Any, List, Set

import numpy as np


def calculate_hit_at_k(predictions: List[Any], relevant_items: Set[Any], k: int = 5) -> float:
    """Returns 1.0 if at least one relevant item appears in top-k predictions, else 0.0."""
    top_k = predictions[:k]
    return 1.0 if any(item in relevant_items for item in top_k) else 0.0


def calculate_mrr_at_k(predictions: List[Any], relevant_items: Set[Any], k: int = 5) -> float:
    """Returns the reciprocal rank (1 / rank) of the first relevant item in top-k, else 0.0."""
    for rank, item in enumerate(predictions[:k], 1):
        if item in relevant_items:
            return 1.0 / rank
    return 0.0


def calculate_ndcg_at_k(predictions: List[Any], relevant_items: Set[Any], k: int = 5) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at rank k (binary relevance).
    DCG = sum((2^rel - 1) / log2(rank + 1))
    """
    top_k = predictions[:k]
    dcg = 0.0
    for rank, item in enumerate(top_k, 1):
        rel = 1.0 if item in relevant_items else 0.0
        dcg += rel / np.log2(rank + 1)

    # Ideal DCG
    ideal_hits = min(len(relevant_items), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
    return float(dcg / idcg) if idcg > 0.0 else 0.0
