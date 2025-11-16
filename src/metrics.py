"""Evaluation metrics for recommendation systems."""

from __future__ import annotations

from typing import List, Set

import numpy as np


def precision_at_k(recommended_items: List[str], relevant_items: Set[str], k: int) -> float:
    """Calculate Precision@K.
    
    Args:
        recommended_items: List of recommended item IDs.
        relevant_items: Set of relevant item IDs.
        k: Number of top recommendations to consider.
        
    Returns:
        Precision@K score.
    """
    if k == 0:
        return 0.0
    
    top_k_recommendations = recommended_items[:k]
    relevant_recommended = sum(1 for item in top_k_recommendations if item in relevant_items)
    
    return relevant_recommended / k


def recall_at_k(recommended_items: List[str], relevant_items: Set[str], k: int) -> float:
    """Calculate Recall@K.
    
    Args:
        recommended_items: List of recommended item IDs.
        relevant_items: Set of relevant item IDs.
        k: Number of top recommendations to consider.
        
    Returns:
        Recall@K score.
    """
    if len(relevant_items) == 0:
        return 0.0
    
    top_k_recommendations = recommended_items[:k]
    relevant_recommended = sum(1 for item in top_k_recommendations if item in relevant_items)
    
    return relevant_recommended / len(relevant_items)


def ndcg_at_k(recommended_items: List[str], relevant_items: Set[str], k: int) -> float:
    """Calculate NDCG@K.
    
    Args:
        recommended_items: List of recommended item IDs.
        relevant_items: Set of relevant item IDs.
        k: Number of top recommendations to consider.
        
    Returns:
        NDCG@K score.
    """
    if k == 0 or len(relevant_items) == 0:
        return 0.0
    
    top_k_recommendations = recommended_items[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(top_k_recommendations):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Calculate IDCG (ideal DCG)
    idcg = 0.0
    for i in range(min(k, len(relevant_items))):
        idcg += 1.0 / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended_items: List[str], relevant_items: Set[str], k: int) -> float:
    """Calculate Hit Rate@K.
    
    Args:
        recommended_items: List of recommended item IDs.
        relevant_items: Set of relevant item IDs.
        k: Number of top recommendations to consider.
        
    Returns:
        Hit Rate@K score (0 or 1).
    """
    top_k_recommendations = recommended_items[:k]
    return 1.0 if any(item in relevant_items for item in top_k_recommendations) else 0.0


def map_at_k(recommended_items: List[str], relevant_items: Set[str], k: int) -> float:
    """Calculate MAP@K (Mean Average Precision).
    
    Args:
        recommended_items: List of recommended item IDs.
        relevant_items: Set of relevant item IDs.
        k: Number of top recommendations to consider.
        
    Returns:
        MAP@K score.
    """
    if len(relevant_items) == 0:
        return 0.0
    
    top_k_recommendations = recommended_items[:k]
    
    precision_sum = 0.0
    relevant_count = 0
    
    for i, item in enumerate(top_k_recommendations):
        if item in relevant_items:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)
    
    return precision_sum / len(relevant_items)


def coverage(recommended_items_all_users: List[List[str]], all_items: Set[str]) -> float:
    """Calculate catalog coverage.
    
    Args:
        recommended_items_all_users: List of recommendations for each user.
        all_items: Set of all available items.
        
    Returns:
        Coverage score (proportion of items recommended).
    """
    recommended_items = set()
    for user_recs in recommended_items_all_users:
        recommended_items.update(user_recs)
    
    return len(recommended_items) / len(all_items) if len(all_items) > 0 else 0.0


def diversity(recommended_items: List[str], item_similarity_matrix: np.ndarray, 
              item_to_idx: dict) -> float:
    """Calculate intra-list diversity.
    
    Args:
        recommended_items: List of recommended item IDs.
        item_similarity_matrix: Item similarity matrix.
        item_to_idx: Mapping from item ID to matrix index.
        
    Returns:
        Average pairwise dissimilarity (1 - similarity).
    """
    if len(recommended_items) < 2:
        return 0.0
    
    similarities = []
    for i in range(len(recommended_items)):
        for j in range(i + 1, len(recommended_items)):
            idx_i = item_to_idx.get(recommended_items[i])
            idx_j = item_to_idx.get(recommended_items[j])
            
            if idx_i is not None and idx_j is not None:
                similarity = item_similarity_matrix[idx_i, idx_j]
                similarities.append(similarity)
    
    if not similarities:
        return 0.0
    
    return 1.0 - np.mean(similarities)
