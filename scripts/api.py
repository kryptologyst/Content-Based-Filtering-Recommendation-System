"""FastAPI service for content-based recommendation system."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from typing import List, Optional
import yaml

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from content_based_recommender import ContentBasedRecommender
from data_utils import DataLoader

# Load configuration
with open("configs/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title="Content-Based Recommendation API",
    description="REST API for content-based filtering recommendations",
    version="1.0.0"
)

# Global variables for models and data
models = {}
items_df = None
interactions_df = None


class RecommendationRequest(BaseModel):
    """Request model for recommendations."""
    user_items: List[str]
    top_k: int = 10
    model_name: str = "TF-IDF"


class SimilarityRequest(BaseModel):
    """Request model for item similarity."""
    item_id: str
    top_k: int = 5
    model_name: str = "TF-IDF"


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    recommendations: List[dict]
    model_used: str


class SimilarityResponse(BaseModel):
    """Response model for item similarity."""
    similar_items: List[dict]
    model_used: str


@app.on_event("startup")
async def startup_event():
    """Initialize models and data on startup."""
    global models, items_df, interactions_df
    
    # Load data
    data_loader = DataLoader(config['data']['data_dir'])
    items_df = data_loader.load_items()
    interactions_df = data_loader.load_interactions()
    
    # Load TF-IDF model
    tfidf_model = ContentBasedRecommender(vectorizer_type="tfidf")
    tfidf_model.fit(items_df, text_column="description")
    models["TF-IDF"] = tfidf_model
    
    # Try to load SBERT model
    try:
        sbert_model = ContentBasedRecommender(vectorizer_type="sbert")
        sbert_model.fit(items_df, text_column="description")
        models["SBERT"] = sbert_model
    except Exception as e:
        print(f"SBERT model not available: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Content-Based Recommendation API",
        "version": "1.0.0",
        "available_models": list(models.keys())
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "models_loaded": len(models)}


@app.get("/items")
async def get_items():
    """Get all items."""
    if items_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    return items_df.to_dict('index')


@app.get("/items/{item_id}")
async def get_item(item_id: str):
    """Get specific item by ID."""
    if items_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    if item_id not in items_df.index:
        raise HTTPException(status_code=404, detail="Item not found")
    
    return items_df.loc[item_id].to_dict()


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get recommendations for a user."""
    if request.model_name not in models:
        raise HTTPException(
            status_code=400, 
            detail=f"Model {request.model_name} not available. Available: {list(models.keys())}"
        )
    
    model = models[request.model_name]
    
    try:
        recommendations = model.recommend_for_user(
            request.user_items, 
            top_k=request.top_k, 
            exclude_seen=True
        )
        
        # Format recommendations with item details
        formatted_recs = []
        for item_id, score in recommendations:
            if item_id in items_df.index:
                item_info = items_df.loc[item_id].to_dict()
                item_info['item_id'] = item_id
                item_info['similarity_score'] = score
                formatted_recs.append(item_info)
        
        return RecommendationResponse(
            recommendations=formatted_recs,
            model_used=request.model_name
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similar", response_model=SimilarityResponse)
async def get_similar_items(request: SimilarityRequest):
    """Get similar items for a given item."""
    if request.model_name not in models:
        raise HTTPException(
            status_code=400, 
            detail=f"Model {request.model_name} not available. Available: {list(models.keys())}"
        )
    
    model = models[request.model_name]
    
    try:
        similar_items = model.get_similar_items(
            request.item_id, 
            top_k=request.top_k
        )
        
        # Format similar items with details
        formatted_items = []
        for item_id, score in similar_items:
            if item_id in items_df.index:
                item_info = items_df.loc[item_id].to_dict()
                item_info['item_id'] = item_id
                item_info['similarity_score'] = score
                formatted_items.append(item_info)
        
        return SimilarityResponse(
            similar_items=formatted_items,
            model_used=request.model_name
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}/interactions")
async def get_user_interactions(user_id: str):
    """Get interaction history for a user."""
    if interactions_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    
    if len(user_interactions) == 0:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Add item details
    interactions_with_details = []
    for _, interaction in user_interactions.iterrows():
        item_id = interaction['item_id']
        if item_id in items_df.index:
            item_info = items_df.loc[item_id].to_dict()
            interaction_dict = interaction.to_dict()
            interaction_dict.update(item_info)
            interactions_with_details.append(interaction_dict)
    
    return interactions_with_details


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
