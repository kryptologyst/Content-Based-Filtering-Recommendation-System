"""Model evaluation and comparison utilities."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .content_based_recommender import ContentBasedRecommender
from .data_utils import DataLoader, get_user_interactions
from .metrics import (
    coverage, diversity, hit_rate_at_k, map_at_k, ndcg_at_k, precision_at_k, recall_at_k
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator for recommendation models."""
    
    def __init__(self, k_values: List[int] = None) -> None:
        """Initialize evaluator.
        
        Args:
            k_values: List of k values for evaluation metrics.
        """
        self.k_values = k_values or [5, 10, 20]
    
    def evaluate_model(
        self,
        model: ContentBasedRecommender,
        test_interactions: pd.DataFrame,
        items_df: pd.DataFrame,
        train_interactions: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Evaluate a single model.
        
        Args:
            model: Trained recommendation model.
            test_interactions: Test set interactions.
            items_df: Items data.
            train_interactions: Training interactions (for coverage calculation).
            
        Returns:
            Dictionary of evaluation metrics.
        """
        metrics = {}
        
        # Get all users in test set
        test_users = test_interactions['user_id'].unique()
        
        # Calculate metrics for each k
        for k in self.k_values:
            precisions = []
            recalls = []
            ndcgs = []
            hit_rates = []
            maps = []
            
            for user_id in test_users:
                # Get user's test items (relevant items)
                user_test_items = set(
                    test_interactions[test_interactions['user_id'] == user_id]['item_id'].tolist()
                )
                
                if len(user_test_items) == 0:
                    continue
                
                # Get user's training items for recommendations
                if train_interactions is not None:
                    user_train_items = get_user_interactions(train_interactions, user_id)
                else:
                    user_train_items = []
                
                # Get recommendations
                if user_train_items:
                    recommendations = model.recommend_for_user(
                        user_train_items, top_k=k, exclude_seen=True
                    )
                    recommended_items = [item_id for item_id, _ in recommendations]
                else:
                    # Cold start: recommend popular items
                    recommended_items = items_df.index.tolist()[:k]
                
                # Calculate metrics
                precisions.append(precision_at_k(recommended_items, user_test_items, k))
                recalls.append(recall_at_k(recommended_items, user_test_items, k))
                ndcgs.append(ndcg_at_k(recommended_items, user_test_items, k))
                hit_rates.append(hit_rate_at_k(recommended_items, user_test_items, k))
                maps.append(map_at_k(recommended_items, user_test_items, k))
            
            # Average metrics across users
            metrics[f'precision@{k}'] = np.mean(precisions) if precisions else 0.0
            metrics[f'recall@{k}'] = np.mean(recalls) if recalls else 0.0
            metrics[f'ndcg@{k}'] = np.mean(ndcgs) if ndcgs else 0.0
            metrics[f'hit_rate@{k}'] = np.mean(hit_rates) if hit_rates else 0.0
            metrics[f'map@{k}'] = np.mean(maps) if maps else 0.0
        
        # Calculate coverage if train interactions provided
        if train_interactions is not None:
            all_recommendations = []
            for user_id in test_users:
                user_train_items = get_user_interactions(train_interactions, user_id)
                if user_train_items:
                    recommendations = model.recommend_for_user(
                        user_train_items, top_k=max(self.k_values), exclude_seen=True
                    )
                    all_recommendations.append([item_id for item_id, _ in recommendations])
            
            if all_recommendations:
                metrics['coverage'] = coverage(all_recommendations, set(items_df.index))
        
        return metrics
    
    def compare_models(
        self,
        models: Dict[str, ContentBasedRecommender],
        test_interactions: pd.DataFrame,
        items_df: pd.DataFrame,
        train_interactions: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Compare multiple models.
        
        Args:
            models: Dictionary of model name -> model instance.
            test_interactions: Test set interactions.
            items_df: Items data.
            train_interactions: Training interactions.
            
        Returns:
            DataFrame with comparison results.
        """
        results = []
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")
            metrics = self.evaluate_model(model, test_interactions, items_df, train_interactions)
            metrics['model'] = model_name
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        return results_df


class BaselineRecommender:
    """Baseline recommendation methods."""
    
    @staticmethod
    def popularity_based(items_df: pd.DataFrame, interactions_df: pd.DataFrame) -> List[str]:
        """Popularity-based recommendations.
        
        Args:
            items_df: Items data.
            interactions_df: Interactions data.
            
        Returns:
            List of item IDs sorted by popularity.
        """
        item_counts = interactions_df['item_id'].value_counts()
        popular_items = item_counts.index.tolist()
        
        # Add items not in interactions
        all_items = set(items_df.index)
        interacted_items = set(item_counts.index)
        non_interacted_items = list(all_items - interacted_items)
        
        return popular_items + non_interacted_items
    
    @staticmethod
    def random_recommendations(items_df: pd.DataFrame, n_recommendations: int = 100) -> List[str]:
        """Random recommendations.
        
        Args:
            items_df: Items data.
            n_recommendations: Number of recommendations.
            
        Returns:
            List of randomly selected item IDs.
        """
        all_items = items_df.index.tolist()
        return np.random.choice(all_items, min(n_recommendations, len(all_items)), replace=False).tolist()


def run_evaluation_experiment(
    data_dir: str = "data",
    k_values: List[int] = None
) -> pd.DataFrame:
    """Run complete evaluation experiment.
    
    Args:
        data_dir: Directory containing data files.
        k_values: List of k values for evaluation.
        
    Returns:
        DataFrame with evaluation results.
    """
    if k_values is None:
        k_values = [5, 10, 20]
    
    logger.info("Starting evaluation experiment")
    
    # Load data
    data_loader = DataLoader(data_dir)
    items_df = data_loader.load_items()
    interactions_df = data_loader.load_interactions()
    
    # Filter cold start users
    interactions_df = filter_cold_start_users(interactions_df, min_interactions=3)
    
    # Split data
    from .data_utils import DataSplitter
    splitter = DataSplitter(test_size=0.2, val_size=0.1)
    train_df, val_df, test_df = splitter.temporal_split(interactions_df)
    
    # Train models
    models = {}
    
    # TF-IDF model
    tfidf_model = ContentBasedRecommender(vectorizer_type="tfidf")
    tfidf_model.fit(items_df, text_column="description")
    models["TF-IDF"] = tfidf_model
    
    # SBERT model (if available)
    try:
        sbert_model = ContentBasedRecommender(vectorizer_type="sbert")
        sbert_model.fit(items_df, text_column="description")
        models["SBERT"] = sbert_model
    except Exception as e:
        logger.warning(f"SBERT model failed to load: {e}")
    
    # Evaluate models
    evaluator = ModelEvaluator(k_values)
    results_df = evaluator.compare_models(models, test_df, items_df, train_df)
    
    # Add baseline results
    baseline_recs = BaselineRecommender.popularity_based(items_df, train_df)
    
    # Create a simple baseline model for evaluation
    class PopularityModel:
        def __init__(self, recommendations: List[str]):
            self.recommendations = recommendations
        
        def recommend_for_user(self, user_items: List[str], top_k: int = 10, exclude_seen: bool = True) -> List[Tuple[str, float]]:
            recs = self.recommendations[:top_k]
            if exclude_seen:
                recs = [item for item in recs if item not in user_items]
            return [(item, 1.0) for item in recs[:top_k]]
    
    popularity_model = PopularityModel(baseline_recs)
    models["Popularity"] = popularity_model
    
    # Evaluate baseline
    baseline_metrics = evaluator.evaluate_model(popularity_model, test_df, items_df, train_df)
    baseline_metrics['model'] = 'Popularity'
    results_df = pd.concat([results_df, pd.DataFrame([baseline_metrics])], ignore_index=True)
    
    logger.info("Evaluation experiment completed")
    return results_df


def filter_cold_start_users(interactions_df: pd.DataFrame, min_interactions: int = 5) -> pd.DataFrame:
    """Filter out users with too few interactions."""
    from .data_utils import filter_cold_start_users as _filter_cold_start_users
    return _filter_cold_start_users(interactions_df, min_interactions)
