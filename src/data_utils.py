"""Data loading and preprocessing utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for recommendation system datasets."""
    
    def __init__(self, data_dir: str = "data") -> None:
        """Initialize data loader.
        
        Args:
            data_dir: Directory containing data files.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def load_items(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load items data.
        
        Args:
            file_path: Path to items CSV file. If None, uses default path.
            
        Returns:
            DataFrame with items data.
        """
        if file_path is None:
            file_path = self.data_dir / "items.csv"
        
        items_df = pd.read_csv(file_path, index_col=0)
        logger.info(f"Loaded {len(items_df)} items from {file_path}")
        return items_df
    
    def load_interactions(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load interactions data.
        
        Args:
            file_path: Path to interactions CSV file. If None, uses default path.
            
        Returns:
            DataFrame with interactions data.
        """
        if file_path is None:
            file_path = self.data_dir / "interactions.csv"
        
        interactions_df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(interactions_df)} interactions from {file_path}")
        return interactions_df
    
    def load_users(self, file_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load users data (optional).
        
        Args:
            file_path: Path to users CSV file. If None, uses default path.
            
        Returns:
            DataFrame with users data or None if file doesn't exist.
        """
        if file_path is None:
            file_path = self.data_dir / "users.csv"
        
        if not Path(file_path).exists():
            logger.info("No users file found, skipping user data")
            return None
        
        users_df = pd.read_csv(file_path, index_col=0)
        logger.info(f"Loaded {len(users_df)} users from {file_path}")
        return users_df


class DataSplitter:
    """Data splitter for train/validation/test sets."""
    
    def __init__(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> None:
        """Initialize data splitter.
        
        Args:
            test_size: Proportion of data for test set.
            val_size: Proportion of data for validation set.
            random_state: Random state for reproducibility.
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
    
    def temporal_split(self, interactions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data temporally (chronological split).
        
        Args:
            interactions_df: DataFrame with interactions data.
            
        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        # Sort by timestamp
        interactions_df = interactions_df.sort_values('timestamp')
        
        # Calculate split indices
        n_total = len(interactions_df)
        test_start = int(n_total * (1 - self.test_size))
        val_start = int(test_start * (1 - self.val_size / (1 - self.test_size)))
        
        train_df = interactions_df.iloc[:val_start].copy()
        val_df = interactions_df.iloc[val_start:test_start].copy()
        test_df = interactions_df.iloc[test_start:].copy()
        
        logger.info(f"Temporal split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        return train_df, val_df, test_df
    
    def random_split(self, interactions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data randomly.
        
        Args:
            interactions_df: DataFrame with interactions data.
            
        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            interactions_df,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # Second split: train vs val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=self.val_size / (1 - self.test_size),
            random_state=self.random_state
        )
        
        logger.info(f"Random split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        return train_df, val_df, test_df
    
    def leave_one_out_split(self, interactions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Leave-one-out split (for each user, keep last interaction for test).
        
        Args:
            interactions_df: DataFrame with interactions data.
            
        Returns:
            Tuple of (train_df, test_df).
        """
        train_interactions = []
        test_interactions = []
        
        for user_id in interactions_df['user_id'].unique():
            user_interactions = interactions_df[interactions_df['user_id'] == user_id].sort_values('timestamp')
            
            if len(user_interactions) > 1:
                train_interactions.append(user_interactions.iloc[:-1])
                test_interactions.append(user_interactions.iloc[-1:])
            else:
                # If user has only one interaction, add to train
                train_interactions.append(user_interactions)
        
        train_df = pd.concat(train_interactions, ignore_index=True)
        test_df = pd.concat(test_interactions, ignore_index=True)
        
        logger.info(f"Leave-one-out split: {len(train_df)} train, {len(test_df)} test")
        return train_df, test_df


def create_user_item_matrix(interactions_df: pd.DataFrame, 
                          items_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, int], Dict[str, int]]:
    """Create user-item interaction matrix.
    
    Args:
        interactions_df: DataFrame with interactions data.
        items_df: DataFrame with items data.
        
    Returns:
        Tuple of (matrix, user_to_idx, item_to_idx).
    """
    # Create mappings
    unique_users = sorted(interactions_df['user_id'].unique())
    unique_items = sorted(items_df.index.unique())
    
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    
    # Create matrix
    matrix = np.zeros((len(unique_users), len(unique_items)))
    
    for _, row in interactions_df.iterrows():
        user_idx = user_to_idx[row['user_id']]
        item_idx = item_to_idx[row['item_id']]
        matrix[user_idx, item_idx] = row.get('rating', 1)  # Default to 1 for implicit feedback
    
    logger.info(f"Created user-item matrix: {matrix.shape}")
    return matrix, user_to_idx, item_to_idx


def get_user_interactions(interactions_df: pd.DataFrame, user_id: str) -> List[str]:
    """Get list of items a user has interacted with.
    
    Args:
        interactions_df: DataFrame with interactions data.
        user_id: User ID.
        
    Returns:
        List of item IDs the user has interacted with.
    """
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    return user_interactions['item_id'].tolist()


def filter_cold_start_users(interactions_df: pd.DataFrame, min_interactions: int = 5) -> pd.DataFrame:
    """Filter out users with too few interactions.
    
    Args:
        interactions_df: DataFrame with interactions data.
        min_interactions: Minimum number of interactions required.
        
    Returns:
        Filtered DataFrame.
    """
    user_counts = interactions_df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    
    filtered_df = interactions_df[interactions_df['user_id'].isin(valid_users)]
    
    logger.info(f"Filtered from {len(interactions_df)} to {len(filtered_df)} interactions "
                f"({len(valid_users)} users with >= {min_interactions} interactions)")
    
    return filtered_df
