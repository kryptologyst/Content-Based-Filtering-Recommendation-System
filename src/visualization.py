"""Visualization utilities for recommendation system results."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Optional

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_metrics_comparison(results_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Plot comparison of metrics across models.
    
    Args:
        results_df: DataFrame with evaluation results.
        save_path: Optional path to save the plot.
    """
    # Get metric columns (exclude 'model' column)
    metric_cols = [col for col in results_df.columns if col != 'model']
    
    # Create subplots
    n_metrics = len(metric_cols)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, metric in enumerate(metric_cols):
        row = i // n_cols
        col = i % n_cols
        
        ax = axes[row, col]
        
        # Create bar plot
        bars = ax.bar(results_df['model'], results_df[metric])
        ax.set_title(f'{metric.replace("@", " @ ").replace("_", " ").title()}')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    # Hide empty subplots
    for i in range(n_metrics, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_metrics_by_k(results_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Plot metrics vs k values.
    
    Args:
        results_df: DataFrame with evaluation results.
        save_path: Optional path to save the plot.
    """
    # Extract k values and metrics
    k_values = []
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    
    for col in results_df.columns:
        if col.startswith('precision@'):
            k = int(col.split('@')[1])
            k_values.append(k)
            precision_scores.append(results_df[col].values)
        elif col.startswith('recall@'):
            recall_scores.append(results_df[col].values)
        elif col.startswith('ndcg@'):
            ndcg_scores.append(results_df[col].values)
    
    if not k_values:
        print("No k-based metrics found in results")
        return
    
    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = results_df['model'].values
    
    # Precision@K
    for i, model in enumerate(models):
        axes[0].plot(k_values, [precision_scores[j][i] for j in range(len(k_values))], 
                    marker='o', label=model)
    axes[0].set_title('Precision@K')
    axes[0].set_xlabel('K')
    axes[0].set_ylabel('Precision')
    axes[0].legend()
    axes[0].grid(True)
    
    # Recall@K
    for i, model in enumerate(models):
        axes[1].plot(k_values, [recall_scores[j][i] for j in range(len(k_values))], 
                    marker='s', label=model)
    axes[1].set_title('Recall@K')
    axes[1].set_xlabel('K')
    axes[1].set_ylabel('Recall')
    axes[1].legend()
    axes[1].grid(True)
    
    # NDCG@K
    for i, model in enumerate(models):
        axes[2].plot(k_values, [ndcg_scores[j][i] for j in range(len(k_values))], 
                    marker='^', label=model)
    axes[2].set_title('NDCG@K')
    axes[2].set_xlabel('K')
    axes[2].set_ylabel('NDCG')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_item_similarity_heatmap(similarity_matrix: np.ndarray, 
                                item_names: List[str],
                                save_path: Optional[str] = None) -> None:
    """Plot item similarity heatmap.
    
    Args:
        similarity_matrix: Item similarity matrix.
        item_names: List of item names.
        save_path: Optional path to save the plot.
    """
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(similarity_matrix, 
                xticklabels=item_names,
                yticklabels=item_names,
                cmap='viridis',
                square=True,
                cbar_kws={'label': 'Cosine Similarity'})
    
    plt.title('Item Similarity Matrix')
    plt.xlabel('Items')
    plt.ylabel('Items')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_genre_distribution(items_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Plot genre distribution.
    
    Args:
        items_df: DataFrame with items data.
        save_path: Optional path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    
    genre_counts = items_df['genre'].value_counts()
    
    # Create bar plot
    bars = plt.bar(range(len(genre_counts)), genre_counts.values)
    plt.xticks(range(len(genre_counts)), genre_counts.index, rotation=45)
    plt.title('Movie Genre Distribution')
    plt.xlabel('Genre')
    plt.ylabel('Number of Movies')
    
    # Add value labels
    for bar, value in zip(bars, genre_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(value), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_user_interaction_stats(interactions_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Plot user interaction statistics.
    
    Args:
        interactions_df: DataFrame with interactions data.
        save_path: Optional path to save the plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # User interaction counts
    user_counts = interactions_df['user_id'].value_counts()
    axes[0, 0].hist(user_counts, bins=20, alpha=0.7)
    axes[0, 0].set_title('Distribution of User Interaction Counts')
    axes[0, 0].set_xlabel('Number of Interactions')
    axes[0, 0].set_ylabel('Number of Users')
    
    # Item popularity
    item_counts = interactions_df['item_id'].value_counts()
    axes[0, 1].hist(item_counts, bins=20, alpha=0.7)
    axes[0, 1].set_title('Distribution of Item Popularity')
    axes[0, 1].set_xlabel('Number of Interactions')
    axes[0, 1].set_ylabel('Number of Items')
    
    # Rating distribution (if available)
    if 'rating' in interactions_df.columns:
        rating_counts = interactions_df['rating'].value_counts().sort_index()
        axes[1, 0].bar(rating_counts.index, rating_counts.values)
        axes[1, 0].set_title('Rating Distribution')
        axes[1, 0].set_xlabel('Rating')
        axes[1, 0].set_ylabel('Number of Interactions')
    
    # Temporal distribution (if timestamp available)
    if 'timestamp' in interactions_df.columns:
        # Convert timestamp to datetime for better visualization
        interactions_df['date'] = pd.to_datetime(interactions_df['timestamp'], unit='s')
        daily_counts = interactions_df.groupby(interactions_df['date'].dt.date).size()
        axes[1, 1].plot(daily_counts.index, daily_counts.values)
        axes[1, 1].set_title('Daily Interaction Counts')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Number of Interactions')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_evaluation_report(results_df: pd.DataFrame, 
                           items_df: pd.DataFrame,
                           interactions_df: pd.DataFrame,
                           output_dir: str = "assets") -> None:
    """Create comprehensive evaluation report with plots.
    
    Args:
        results_df: DataFrame with evaluation results.
        items_df: DataFrame with items data.
        interactions_df: DataFrame with interactions data.
        output_dir: Directory to save plots.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot metrics comparison
    plot_metrics_comparison(results_df, f"{output_dir}/metrics_comparison.png")
    
    # Plot metrics by k
    plot_metrics_by_k(results_df, f"{output_dir}/metrics_by_k.png")
    
    # Plot genre distribution
    plot_genre_distribution(items_df, f"{output_dir}/genre_distribution.png")
    
    # Plot user interaction stats
    plot_user_interaction_stats(interactions_df, f"{output_dir}/user_interaction_stats.png")
    
    print(f"Evaluation report saved to {output_dir}/")
