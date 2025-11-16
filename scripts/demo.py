"""Streamlit demo for content-based filtering recommender."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from content_based_recommender import ContentBasedRecommender
from data_utils import DataLoader, get_user_interactions


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_models_and_data(config: dict) -> tuple:
    """Load trained models and data."""
    data_loader = DataLoader(config['data']['data_dir'])
    items_df = data_loader.load_items()
    interactions_df = data_loader.load_interactions()
    
    # Load TF-IDF model
    tfidf_model = ContentBasedRecommender(vectorizer_type="tfidf")
    tfidf_model.fit(items_df, text_column="description")
    
    # Try to load SBERT model
    sbert_model = None
    try:
        sbert_model = ContentBasedRecommender(vectorizer_type="sbert")
        sbert_model.fit(items_df, text_column="description")
    except Exception as e:
        st.warning(f"SBERT model not available: {e}")
    
    return items_df, interactions_df, tfidf_model, sbert_model


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Content-Based Filtering Demo",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Content-Based Movie Recommendation System")
    st.markdown("""
    This demo showcases a content-based filtering recommendation system using TF-IDF and 
    sentence transformers to recommend movies based on their descriptions and features.
    """)
    
    # Load configuration
    try:
        config = load_config()
    except FileNotFoundError:
        st.error("Configuration file not found. Please run the training script first.")
        return
    
    # Load models and data
    with st.spinner("Loading models and data..."):
        try:
            items_df, interactions_df, tfidf_model, sbert_model = load_models_and_data(config)
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            return
    
    st.success("Models loaded successfully!")
    
    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    available_models = {"TF-IDF": tfidf_model}
    if sbert_model is not None:
        available_models["SBERT"] = sbert_model
    
    selected_model_name = st.sidebar.selectbox(
        "Choose recommendation model:",
        list(available_models.keys())
    )
    selected_model = available_models[selected_model_name]
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🎯 User Recommendations", "🔍 Item Similarity", "📊 Data Overview"])
    
    with tab1:
        st.header("Get Recommendations for a User")
        
        # User selection
        user_ids = sorted(interactions_df['user_id'].unique())
        selected_user = st.selectbox("Select a user:", user_ids)
        
        if selected_user:
            # Get user's interaction history
            user_items = get_user_interactions(interactions_df, selected_user)
            
            st.subheader(f"User {selected_user}'s Movie History")
            if user_items:
                user_movies = items_df.loc[user_items, ['title', 'genre', 'year']]
                st.dataframe(user_movies, use_container_width=True)
                
                # Get recommendations
                top_k = st.slider("Number of recommendations:", 5, 20, 10)
                
                if st.button("Get Recommendations"):
                    with st.spinner("Generating recommendations..."):
                        recommendations = selected_model.recommend_for_user(
                            user_items, top_k=top_k, exclude_seen=True
                        )
                    
                    st.subheader(f"Top {top_k} Recommendations")
                    
                    for i, (item_id, score) in enumerate(recommendations, 1):
                        movie_info = items_df.loc[item_id]
                        
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**{i}. {movie_info['title']}** ({movie_info['year']})")
                            st.write(f"*{movie_info['genre']}*")
                            st.write(f"{movie_info['description']}")
                        
                        with col2:
                            st.metric("Similarity Score", f"{score:.3f}")
                        
                        st.divider()
            else:
                st.warning("This user has no interaction history.")
    
    with tab2:
        st.header("Find Similar Movies")
        
        # Movie selection
        movie_titles = items_df['title'].tolist()
        selected_movie_title = st.selectbox("Select a movie:", movie_titles)
        
        if selected_movie_title:
            selected_movie_id = items_df[items_df['title'] == selected_movie_title].index[0]
            movie_info = items_df.loc[selected_movie_id]
            
            st.subheader(f"Selected Movie: {movie_info['title']}")
            st.write(f"**Genre:** {movie_info['genre']}")
            st.write(f"**Year:** {movie_info['year']}")
            st.write(f"**Description:** {movie_info['description']}")
            
            # Get similar movies
            top_k = st.slider("Number of similar movies:", 5, 15, 8)
            
            if st.button("Find Similar Movies"):
                with st.spinner("Finding similar movies..."):
                    similar_movies = selected_model.get_similar_items(
                        selected_movie_id, top_k=top_k
                    )
                
                st.subheader(f"Top {top_k} Similar Movies")
                
                for i, (item_id, score) in enumerate(similar_movies, 1):
                    similar_movie = items_df.loc[item_id]
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**{i}. {similar_movie['title']}** ({similar_movie['year']})")
                        st.write(f"*{similar_movie['genre']}*")
                        st.write(f"{similar_movie['description']}")
                    
                    with col2:
                        st.metric("Similarity Score", f"{score:.3f}")
                    
                    st.divider()
    
    with tab3:
        st.header("Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Movies", len(items_df))
        
        with col2:
            st.metric("Total Users", len(interactions_df['user_id'].unique()))
        
        with col3:
            st.metric("Total Interactions", len(interactions_df))
        
        st.subheader("Movie Genres Distribution")
        genre_counts = items_df['genre'].value_counts()
        st.bar_chart(genre_counts)
        
        st.subheader("Sample Movies")
        st.dataframe(
            items_df[['title', 'genre', 'year', 'description']].head(10),
            use_container_width=True
        )
        
        st.subheader("User Interaction Statistics")
        user_stats = interactions_df.groupby('user_id').size().describe()
        st.dataframe(user_stats)


if __name__ == "__main__":
    main()
