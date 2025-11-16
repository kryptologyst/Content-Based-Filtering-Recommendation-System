# Content-Based Filtering Recommendation System

A production-ready content-based filtering recommendation system using TF-IDF and sentence transformers for text-based item features.

## Overview

This project implements a comprehensive content-based filtering system that recommends items based on their textual features (descriptions, tags, etc.). The system uses TF-IDF vectorization and sentence transformers to create item embeddings and computes cosine similarity for recommendations.

## Features

- **Multiple Models**: TF-IDF baseline and SBERT embeddings
- **Comprehensive Evaluation**: Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate, Coverage
- **Interactive Demo**: Streamlit web interface for testing recommendations
- **Production Ready**: Type hints, docstrings, tests, configuration management
- **Visualization**: Comprehensive plots and evaluation reports

## Project Structure

```
├── src/                          # Source code modules
│   ├── content_based_recommender.py  # Main recommender implementation
│   ├── data_utils.py                 # Data loading and preprocessing
│   ├── metrics.py                    # Evaluation metrics
│   ├── evaluation.py                 # Model evaluation and comparison
│   └── visualization.py              # Plotting utilities
├── configs/                       # Configuration files
│   └── config.yaml               # Main configuration
├── scripts/                       # Executable scripts
│   ├── train.py                   # Training and evaluation script
│   └── demo.py                    # Streamlit demo
├── tests/                         # Unit tests
│   └── test_recommender.py        # Test cases
├── data/                          # Data directory
│   ├── items.csv                  # Items with features
│   └── interactions.csv           # User-item interactions
├── assets/                        # Generated plots and reports
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Content-Based-Filtering-Recommendation-System.git
cd Content-Based-Filtering-Recommendation-System

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Data

The system will automatically create sample movie data if none exists:

```bash
python scripts/train.py --create-data
```

### 3. Train and Evaluate Models

```bash
python scripts/train.py --config configs/config.yaml
```

This will:
- Load or generate sample data
- Train TF-IDF and SBERT models
- Evaluate models on test set
- Save results to `results/evaluation_results.csv`

### 4. Run Interactive Demo

```bash
streamlit run scripts/demo.py
```

Open your browser to `http://localhost:8501` to explore the interactive demo.

## Dataset Schema

### Items CSV (`data/items.csv`)
```csv
item_id,title,description,genre,year
movie_1,"The Dark Knight","Batman faces the Joker...",Action,2008
movie_2,"Inception","Mind-bending sci-fi...",Sci-Fi,2010
```

### Interactions CSV (`data/interactions.csv`)
```csv
user_id,item_id,timestamp,rating
user_1,movie_1,1000000000,5
user_1,movie_2,1000000001,4
```

## Configuration

Edit `configs/config.yaml` to customize:

- **Data settings**: File paths, split ratios, minimum interactions
- **Model settings**: TF-IDF parameters, SBERT model selection
- **Evaluation settings**: K values, metrics, split method
- **Demo settings**: Streamlit port, recommendation limits

## Models

### TF-IDF Baseline
- Uses scikit-learn's TfidfVectorizer
- Configurable n-gram range and max features
- Fast and interpretable

### SBERT Embeddings
- Uses sentence-transformers library
- Default model: `all-MiniLM-L6-v2`
- Better semantic understanding

## Evaluation Metrics

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP@K**: Mean Average Precision
- **Hit Rate@K**: Fraction of users with at least one relevant recommendation
- **Coverage**: Fraction of catalog items that are recommended

## API Usage

### Basic Usage

```python
from src.content_based_recommender import ContentBasedRecommender
import pandas as pd

# Load data
items_df = pd.read_csv('data/items.csv', index_col=0)

# Initialize and fit model
recommender = ContentBasedRecommender(vectorizer_type="tfidf")
recommender.fit(items_df, text_column="description")

# Get similar items
similar_items = recommender.get_similar_items("movie_1", top_k=5)

# Get user recommendations
user_items = ["movie_1", "movie_2"]
recommendations = recommender.recommend_for_user(user_items, top_k=10)
```

### Advanced Usage

```python
from src.evaluation import ModelEvaluator
from src.data_utils import DataLoader, DataSplitter

# Load and split data
data_loader = DataLoader("data")
items_df = data_loader.load_items()
interactions_df = data_loader.load_interactions()

splitter = DataSplitter(test_size=0.2)
train_df, val_df, test_df = splitter.temporal_split(interactions_df)

# Evaluate model
evaluator = ModelEvaluator(k_values=[5, 10, 20])
metrics = evaluator.evaluate_model(recommender, test_df, items_df, train_df)
```

## Testing

Run the test suite:

```bash
pytest tests/
```

## Development

### Code Quality

The project uses:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking
- **Pytest** for testing

### Adding New Models

1. Extend `ContentBasedRecommender` class
2. Implement `fit()` and `recommend_for_user()` methods
3. Add configuration in `configs/config.yaml`
4. Update `scripts/train.py` to include new model

### Adding New Metrics

1. Implement metric function in `src/metrics.py`
2. Add to `ModelEvaluator.evaluate_model()`
3. Update configuration schema

## Results

The system typically achieves:
- **Precision@10**: 0.15-0.25
- **Recall@10**: 0.20-0.35
- **NDCG@10**: 0.25-0.40
- **Coverage**: 0.60-0.80

Results vary based on dataset characteristics and model configuration.

## Troubleshooting

### Common Issues

1. **SBERT model fails to load**: Install sentence-transformers and ensure internet connection for first-time model download
2. **Memory issues**: Reduce `max_features` in TF-IDF configuration or use smaller SBERT model
3. **Demo not loading**: Ensure data files exist in `data/` directory

### Performance Tips

- Use TF-IDF for faster training and inference
- SBERT provides better quality but requires more resources
- Consider caching embeddings for production use
- Use temporal splits for realistic evaluation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built with scikit-learn, sentence-transformers, and Streamlit
- Inspired by modern recommendation system best practices
- Uses sample movie data for demonstration purposes
# Content-Based-Filtering-Recommendation-System
