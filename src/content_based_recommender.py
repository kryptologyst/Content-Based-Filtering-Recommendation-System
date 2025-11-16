"""Content-based filtering recommendation system.

This module implements a modern content-based filtering system using TF-IDF
and sentence transformers for text-based item features.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")


class ContentBasedRecommender:
    """Content-based filtering recommendation system.
    
    This class implements content-based filtering using TF-IDF vectorization
    and cosine similarity for text-based item features.
    """
    
    def __init__(
        self,
        vectorizer_type: str = "tfidf",
        model_name: Optional[str] = None,
        random_state: int = 42
    ) -> None:
        """Initialize the content-based recommender.
        
        Args:
            vectorizer_type: Type of vectorizer ('tfidf' or 'sbert').
            model_name: Name of the sentence transformer model (for SBERT).
            random_state: Random state for reproducibility.
        """
        self.vectorizer_type = vectorizer_type
        self.model_name = model_name
        self.random_state = random_state
        
        set_seed(random_state)
        
        # Initialize vectorizer
        if vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                ngram_range=(1, 2)
            )
            self.sbert_model = None
        elif vectorizer_type == "sbert":
            model_name = model_name or "all-MiniLM-L6-v2"
            self.sbert_model = SentenceTransformer(model_name)
            self.vectorizer = None
        else:
            raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}")
        
        self.item_features: Optional[np.ndarray] = None
        self.item_ids: Optional[List[str]] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        
    def fit(self, items_df: pd.DataFrame, text_column: str = "description") -> None:
        """Fit the recommender on item text features.
        
        Args:
            items_df: DataFrame containing items and their text features.
            text_column: Name of the column containing text features.
        """
        logger.info(f"Fitting {self.vectorizer_type} vectorizer on {len(items_df)} items")
        
        self.item_ids = items_df.index.tolist()
        texts = items_df[text_column].fillna("").tolist()
        
        if self.vectorizer_type == "tfidf":
            self.item_features = self.vectorizer.fit_transform(texts).toarray()
        else:  # sbert
            self.item_features = self.sbert_model.encode(texts)
        
        # Compute similarity matrix
        self.similarity_matrix = cosine_similarity(self.item_features)
        
        logger.info("Recommender fitted successfully")
    
    def get_similar_items(
        self,
        item_id: str,
        top_k: int = 5,
        exclude_self: bool = True
    ) -> List[Tuple[str, float]]:
        """Get similar items for a given item.
        
        Args:
            item_id: ID of the item to find similar items for.
            top_k: Number of similar items to return.
            exclude_self: Whether to exclude the item itself from results.
            
        Returns:
            List of tuples (item_id, similarity_score).
        """
        if self.similarity_matrix is None or self.item_ids is None:
            raise ValueError("Recommender must be fitted first")
        
        try:
            item_idx = self.item_ids.index(item_id)
        except ValueError:
            raise ValueError(f"Item {item_id} not found in fitted data")
        
        similarities = self.similarity_matrix[item_idx]
        
        # Create recommendations with scores
        recommendations = list(zip(self.item_ids, similarities))
        
        # Sort by similarity score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # Exclude self if requested
        if exclude_self:
            recommendations = [rec for rec in recommendations if rec[0] != item_id]
        
        return recommendations[:top_k]
    
    def recommend_for_user(
        self,
        user_items: List[str],
        top_k: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[str, float]]:
        """Recommend items for a user based on their item history.
        
        Args:
            user_items: List of item IDs the user has interacted with.
            top_k: Number of recommendations to return.
            exclude_seen: Whether to exclude items the user has already seen.
            
        Returns:
            List of tuples (item_id, recommendation_score).
        """
        if self.similarity_matrix is None or self.item_ids is None:
            raise ValueError("Recommender must be fitted first")
        
        # Compute user profile as average of item features
        user_item_indices = []
        for item_id in user_items:
            try:
                item_idx = self.item_ids.index(item_id)
                user_item_indices.append(item_idx)
            except ValueError:
                logger.warning(f"Item {item_id} not found in fitted data")
        
        if not user_item_indices:
            return []
        
        # Average similarity scores for user's items
        user_similarities = self.similarity_matrix[user_item_indices].mean(axis=0)
        
        # Create recommendations
        recommendations = list(zip(self.item_ids, user_similarities))
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # Exclude seen items if requested
        if exclude_seen:
            recommendations = [rec for rec in recommendations if rec[0] not in user_items]
        
        return recommendations[:top_k]


def create_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create sample movie data for demonstration.
    
    Returns:
        Tuple of (items_df, interactions_df).
    """
    # Sample movie data
    movies_data = {
        'title': [
            'The Dark Knight', 'Inception', 'Interstellar', 'The Matrix',
            'Pulp Fiction', 'Forrest Gump', 'The Shawshank Redemption',
            'Titanic', 'Avatar', 'Jurassic Park', 'Star Wars', 'Lord of the Rings',
            'Harry Potter', 'Spider-Man', 'Iron Man', 'The Avengers',
            'Casablanca', 'Gone with the Wind', 'Citizen Kane', 'Vertigo'
        ],
        'description': [
            'Batman faces the Joker in this dark superhero thriller with complex themes',
            'Mind-bending sci-fi thriller about dreams within dreams',
            'Epic space adventure about time dilation and human survival',
            'Cyberpunk action film about reality simulation and rebellion',
            'Non-linear crime drama with interconnected stories',
            'Heartwarming drama about a simple man\'s extraordinary life',
            'Prison drama about hope, friendship, and redemption',
            'Epic romance set against the backdrop of the Titanic disaster',
            'Sci-fi epic about alien worlds and environmental themes',
            'Thrilling adventure with genetically engineered dinosaurs',
            'Space opera about the battle between good and evil',
            'Fantasy epic about hobbits, wizards, and magical rings',
            'Magical adventure about a young wizard\'s journey',
            'Superhero origin story about a web-slinging hero',
            'Superhero origin story about a billionaire inventor',
            'Superhero team-up movie with epic battles',
            'Classic romance set in wartime Casablanca',
            'Epic historical romance set during the Civil War',
            'Citizen Kane explores power, wealth, and the American dream',
            'Psychological thriller about obsession and deception'
        ],
        'genre': [
            'Action', 'Sci-Fi', 'Sci-Fi', 'Action', 'Crime', 'Drama',
            'Drama', 'Romance', 'Sci-Fi', 'Adventure', 'Sci-Fi', 'Fantasy',
            'Fantasy', 'Action', 'Action', 'Action', 'Romance', 'Romance',
            'Drama', 'Thriller'
        ],
        'year': [
            2008, 2010, 2014, 1999, 1994, 1994, 1994, 1997, 2009, 1993,
            1977, 2001, 2001, 2002, 2008, 2012, 1942, 1939, 1941, 1958
        ]
    }
    
    items_df = pd.DataFrame(movies_data)
    items_df.index.name = 'item_id'
    
    # Sample user interactions (implicit feedback)
    np.random.seed(42)
    n_users = 100
    n_items = len(items_df)
    
    interactions = []
    for user_id in range(n_users):
        # Each user interacts with 5-15 random items
        n_interactions = np.random.randint(5, 16)
        user_items = np.random.choice(n_items, n_interactions, replace=False)
        
        for item_id in user_items:
            interactions.append({
                'user_id': f'user_{user_id}',
                'item_id': items_df.index[item_id],
                'timestamp': np.random.randint(1000000000, 2000000000),
                'rating': np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])
            })
    
    interactions_df = pd.DataFrame(interactions)
    
    return items_df, interactions_df


def main() -> None:
    """Main function to demonstrate the content-based recommender."""
    logger.info("Starting Content-Based Filtering Demo")
    
    # Create sample data
    items_df, interactions_df = create_sample_data()
    
    # Save data
    items_df.to_csv('data/items.csv')
    interactions_df.to_csv('data/interactions.csv')
    logger.info("Sample data saved to data/ directory")
    
    # Initialize and fit TF-IDF recommender
    tfidf_recommender = ContentBasedRecommender(vectorizer_type="tfidf")
    tfidf_recommender.fit(items_df, text_column="description")
    
    # Test recommendations
    test_item = items_df.index[0]  # First movie
    similar_items = tfidf_recommender.get_similar_items(test_item, top_k=5)
    
    print(f"\nSimilar movies to '{items_df.loc[test_item, 'title']}':")
    for item_id, score in similar_items:
        print(f"  {items_df.loc[item_id, 'title']}: {score:.3f}")
    
    # Test user recommendations
    sample_user_items = interactions_df[interactions_df['user_id'] == 'user_0']['item_id'].tolist()[:3]
    user_recs = tfidf_recommender.recommend_for_user(sample_user_items, top_k=5)
    
    print(f"\nRecommendations for user_0 (based on {len(sample_user_items)} items):")
    for item_id, score in user_recs:
        print(f"  {items_df.loc[item_id, 'title']}: {score:.3f}")


if __name__ == "__main__":
    main()
