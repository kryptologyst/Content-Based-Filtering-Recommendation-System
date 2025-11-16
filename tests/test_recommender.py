"""Tests for content-based recommendation system."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from content_based_recommender import ContentBasedRecommender, set_seed
from data_utils import DataLoader, DataSplitter, create_user_item_matrix
from metrics import precision_at_k, recall_at_k, ndcg_at_k


class TestContentBasedRecommender:
    """Test cases for ContentBasedRecommender."""
    
    def setup_method(self):
        """Set up test data."""
        set_seed(42)
        
        # Create sample data
        self.items_data = {
            'title': ['Movie1', 'Movie2', 'Movie3', 'Movie4', 'Movie5'],
            'description': [
                'Action movie with thrilling sequences',
                'Romantic comedy with funny moments',
                'Action-packed superhero movie',
                'Drama about love and loss',
                'A romantic comedy with unexpected twists'
            ],
            'genre': ['Action', 'Romance', 'Action', 'Drama', 'Romance']
        }
        self.items_df = pd.DataFrame(self.items_data)
        self.items_df.index.name = 'item_id'
        
        # Create sample interactions
        self.interactions_data = {
            'user_id': ['user1', 'user1', 'user2', 'user2', 'user3'],
            'item_id': ['Movie1', 'Movie2', 'Movie1', 'Movie3', 'Movie4'],
            'rating': [5, 4, 3, 5, 4],
            'timestamp': [1000000000, 1000000001, 1000000002, 1000000003, 1000000004]
        }
        self.interactions_df = pd.DataFrame(self.interactions_data)
    
    def test_tfidf_recommender_initialization(self):
        """Test TF-IDF recommender initialization."""
        recommender = ContentBasedRecommender(vectorizer_type="tfidf")
        assert recommender.vectorizer_type == "tfidf"
        assert recommender.vectorizer is not None
        assert recommender.sbert_model is None
    
    def test_sbert_recommender_initialization(self):
        """Test SBERT recommender initialization."""
        recommender = ContentBasedRecommender(vectorizer_type="sbert")
        assert recommender.vectorizer_type == "sbert"
        assert recommender.vectorizer is None
        assert recommender.sbert_model is not None
    
    def test_fit_tfidf_recommender(self):
        """Test fitting TF-IDF recommender."""
        recommender = ContentBasedRecommender(vectorizer_type="tfidf")
        recommender.fit(self.items_df, text_column="description")
        
        assert recommender.item_features is not None
        assert recommender.item_ids is not None
        assert recommender.similarity_matrix is not None
        assert len(recommender.item_ids) == len(self.items_df)
        assert recommender.similarity_matrix.shape == (len(self.items_df), len(self.items_df))
    
    def test_get_similar_items(self):
        """Test getting similar items."""
        recommender = ContentBasedRecommender(vectorizer_type="tfidf")
        recommender.fit(self.items_df, text_column="description")
        
        similar_items = recommender.get_similar_items('Movie1', top_k=3)
        
        assert len(similar_items) == 3
        assert all(isinstance(item_id, str) for item_id, _ in similar_items)
        assert all(isinstance(score, float) for _, score in similar_items)
        assert all(0 <= score <= 1 for _, score in similar_items)
    
    def test_recommend_for_user(self):
        """Test user recommendations."""
        recommender = ContentBasedRecommender(vectorizer_type="tfidf")
        recommender.fit(self.items_df, text_column="description")
        
        user_items = ['Movie1', 'Movie2']
        recommendations = recommender.recommend_for_user(user_items, top_k=3)
        
        assert len(recommendations) == 3
        assert all(isinstance(item_id, str) for item_id, _ in recommendations)
        assert all(isinstance(score, float) for _, score in recommendations)
    
    def test_recommend_for_user_exclude_seen(self):
        """Test user recommendations with exclude_seen=True."""
        recommender = ContentBasedRecommender(vectorizer_type="tfidf")
        recommender.fit(self.items_df, text_column="description")
        
        user_items = ['Movie1', 'Movie2']
        recommendations = recommender.recommend_for_user(user_items, top_k=5, exclude_seen=True)
        
        recommended_item_ids = [item_id for item_id, _ in recommendations]
        assert 'Movie1' not in recommended_item_ids
        assert 'Movie2' not in recommended_item_ids


class TestMetrics:
    """Test cases for evaluation metrics."""
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        recommended_items = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant_items = {'item1', 'item3', 'item5'}
        
        precision = precision_at_k(recommended_items, relevant_items, k=5)
        expected = 3 / 5  # 3 relevant items out of 5 recommendations
        assert abs(precision - expected) < 1e-6
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        recommended_items = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant_items = {'item1', 'item3', 'item5', 'item6'}
        
        recall = recall_at_k(recommended_items, relevant_items, k=5)
        expected = 3 / 4  # 3 relevant items found out of 4 total relevant
        assert abs(recall - expected) < 1e-6
    
    def test_ndcg_at_k(self):
        """Test NDCG@k calculation."""
        recommended_items = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant_items = {'item1', 'item3', 'item5'}
        
        ndcg = ndcg_at_k(recommended_items, relevant_items, k=5)
        assert 0 <= ndcg <= 1
    
    def test_precision_at_k_zero_k(self):
        """Test precision@k with k=0."""
        recommended_items = ['item1', 'item2', 'item3']
        relevant_items = {'item1', 'item2'}
        
        precision = precision_at_k(recommended_items, relevant_items, k=0)
        assert precision == 0.0
    
    def test_recall_at_k_empty_relevant(self):
        """Test recall@k with empty relevant items."""
        recommended_items = ['item1', 'item2', 'item3']
        relevant_items = set()
        
        recall = recall_at_k(recommended_items, relevant_items, k=3)
        assert recall == 0.0


class TestDataUtils:
    """Test cases for data utilities."""
    
    def test_data_splitter_temporal_split(self):
        """Test temporal data splitting."""
        # Create sample data with timestamps
        data = {
            'user_id': ['user1'] * 10,
            'item_id': [f'item{i}' for i in range(10)],
            'timestamp': list(range(1000000000, 1000000010))
        }
        df = pd.DataFrame(data)
        
        splitter = DataSplitter(test_size=0.3, val_size=0.2)
        train_df, val_df, test_df = splitter.temporal_split(df)
        
        assert len(train_df) == 5  # 50% for train
        assert len(val_df) == 2    # 20% for val
        assert len(test_df) == 3   # 30% for test
        assert len(train_df) + len(val_df) + len(test_df) == len(df)
    
    def test_data_splitter_random_split(self):
        """Test random data splitting."""
        # Create sample data
        data = {
            'user_id': ['user1'] * 10,
            'item_id': [f'item{i}' for i in range(10)],
            'timestamp': list(range(1000000000, 1000000010))
        }
        df = pd.DataFrame(data)
        
        splitter = DataSplitter(test_size=0.3, val_size=0.2, random_state=42)
        train_df, val_df, test_df = splitter.random_split(df)
        
        assert len(train_df) + len(val_df) + len(test_df) == len(df)
        assert len(test_df) == 3  # 30% for test
    
    def test_create_user_item_matrix(self):
        """Test user-item matrix creation."""
        items_data = {
            'title': ['Movie1', 'Movie2', 'Movie3'],
            'description': ['desc1', 'desc2', 'desc3']
        }
        items_df = pd.DataFrame(items_data)
        items_df.index = ['item1', 'item2', 'item3']
        
        interactions_data = {
            'user_id': ['user1', 'user1', 'user2'],
            'item_id': ['item1', 'item2', 'item1'],
            'rating': [5, 4, 3]
        }
        interactions_df = pd.DataFrame(interactions_data)
        
        matrix, user_to_idx, item_to_idx = create_user_item_matrix(interactions_df, items_df)
        
        assert matrix.shape == (2, 3)  # 2 users, 3 items
        assert len(user_to_idx) == 2
        assert len(item_to_idx) == 3
        assert matrix[0, 0] == 5  # user1, item1
        assert matrix[0, 1] == 4  # user1, item2
        assert matrix[1, 0] == 3  # user2, item1


if __name__ == "__main__":
    pytest.main([__file__])
