"""Main training and evaluation script."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from content_based_recommender import ContentBasedRecommender, create_sample_data
from data_utils import DataLoader, DataSplitter, filter_cold_start_users
from evaluation import ModelEvaluator, run_evaluation_experiment

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_data_if_missing(data_dir: str) -> None:
    """Create sample data if it doesn't exist.
    
    Args:
        data_dir: Directory to create data in.
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    items_file = data_path / "items.csv"
    interactions_file = data_path / "interactions.csv"
    
    if not items_file.exists() or not interactions_file.exists():
        logger.info("Creating sample data...")
        items_df, interactions_df = create_sample_data()
        
        items_df.to_csv(items_file)
        interactions_df.to_csv(interactions_file)
        
        logger.info(f"Sample data created in {data_dir}")


def train_and_evaluate(config: dict) -> pd.DataFrame:
    """Train models and evaluate them.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        DataFrame with evaluation results.
    """
    data_config = config['data']
    
    # Create data if missing
    create_data_if_missing(data_config['data_dir'])
    
    # Load data
    data_loader = DataLoader(data_config['data_dir'])
    items_df = data_loader.load_items()
    interactions_df = data_loader.load_interactions()
    
    # Filter cold start users
    interactions_df = filter_cold_start_users(
        interactions_df, 
        min_interactions=data_config['min_user_interactions']
    )
    
    # Split data
    splitter = DataSplitter(
        test_size=data_config['test_size'],
        val_size=data_config['val_size'],
        random_state=data_config['random_state']
    )
    
    if config['evaluation']['split_method'] == 'temporal':
        train_df, val_df, test_df = splitter.temporal_split(interactions_df)
    else:
        train_df, val_df, test_df = splitter.random_split(interactions_df)
    
    # Train models
    models = {}
    model_configs = config['models']
    
    # TF-IDF model
    if 'tfidf' in model_configs:
        tfidf_config = model_configs['tfidf']
        tfidf_model = ContentBasedRecommender(
            vectorizer_type=tfidf_config['vectorizer_type']
        )
        tfidf_model.fit(items_df, text_column="description")
        models["TF-IDF"] = tfidf_model
        logger.info("TF-IDF model trained")
    
    # SBERT model
    if 'sbert' in model_configs:
        try:
            sbert_config = model_configs['sbert']
            sbert_model = ContentBasedRecommender(
                vectorizer_type=sbert_config['vectorizer_type'],
                model_name=sbert_config['model_name']
            )
            sbert_model.fit(items_df, text_column="description")
            models["SBERT"] = sbert_model
            logger.info("SBERT model trained")
        except Exception as e:
            logger.warning(f"SBERT model failed to train: {e}")
    
    # Evaluate models
    evaluator = ModelEvaluator(k_values=config['evaluation']['k_values'])
    results_df = evaluator.compare_models(models, test_df, items_df, train_df)
    
    return results_df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Content-based filtering experiment")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="results/evaluation_results.csv",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--create-data", 
        action="store_true",
        help="Create sample data if it doesn't exist"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run experiment
    logger.info("Starting content-based filtering experiment")
    results_df = train_and_evaluate(config)
    
    # Save results
    results_df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(results_df.to_string(index=False))
    
    # Print best model for each metric
    print("\n" + "="*50)
    print("BEST MODELS BY METRIC")
    print("="*50)
    
    metric_cols = [col for col in results_df.columns if col != 'model']
    for metric in metric_cols:
        if metric in results_df.columns:
            best_idx = results_df[metric].idxmax()
            best_model = results_df.loc[best_idx, 'model']
            best_score = results_df.loc[best_idx, metric]
            print(f"{metric}: {best_model} ({best_score:.4f})")


if __name__ == "__main__":
    main()
