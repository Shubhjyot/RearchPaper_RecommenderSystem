import os
import pandas as pd
import numpy as np
import time
from data_preprocessing import DataPreprocessor
from content_based_recommender import ContentBasedRecommender
from collaborative_recommender import CollaborativeRecommender
from hybrid_recommender import HybridRecommender
from conversational_rag import ConversationalRAG

def main():
    """Build all models for the paper recommender system."""
    print("Starting model building process...")
    start_time = time.time()
    
    # Step 1: Preprocess data
    print("\n=== Step 1: Preprocessing Data ===")
    preprocessor = DataPreprocessor("arxiv_papers_unique_20250425_Final.csv")
    processed_data = preprocessor.preprocess_data()
    preprocessor.save_processed_data()
    
    # Step 2: Build content-based models
    print("\n=== Step 2: Building Content-Based Models ===")
    content_recommender = ContentBasedRecommender(processed_data)
    content_recommender.build_tfidf_model()
    content_recommender.build_transformer_model()
    content_recommender.build_faiss_index()
    content_recommender.save_models()
    
    # Step 3: Build collaborative filtering models
    print("\n=== Step 3: Building Collaborative Filtering Models ===")
    collaborative_recommender = CollaborativeRecommender(processed_data)
    collaborative_recommender.simulate_user_data(n_users=500)  # Use fewer users for faster training
    collaborative_recommender.prepare_surprise_data()
    collaborative_recommender.train_svd_model(n_factors=50, n_epochs=10)  # Reduced factors and epochs for faster training
    collaborative_recommender.train_svdpp_model(n_factors=50, n_epochs=10)  # Reduced factors and epochs for faster training
    collaborative_recommender.save_models()
    
    # Step 4: Build RAG models
    print("\n=== Step 4: Building RAG Models ===")
    rag_system = ConversationalRAG(processed_data)
    rag_system.build_vector_db()
    rag_system.save_models()
    
    # Done
    total_time = time.time() - start_time
    print(f"\nAll models built successfully in {total_time:.2f} seconds ({total_time/60:.2f} minutes)!")
    print("You can now run the Streamlit app with: streamlit run app.py")

if __name__ == "__main__":
    main()
