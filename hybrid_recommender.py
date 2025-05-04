import pandas as pd
import numpy as np
from content_based_recommender import ContentBasedRecommender
from collaborative_recommender import CollaborativeRecommender
import joblib
import os

class HybridRecommender:
    def __init__(self, processed_data=None, data_path='processed_data.pkl'):
        """Initialize the hybrid recommender with processed data."""
        if processed_data is not None:
            self.data = processed_data
        elif os.path.exists(data_path):
            self.data = joblib.load(data_path)
        else:
            raise FileNotFoundError(f"Processed data file {data_path} not found.")
        
        # Initialize the component recommenders
        self.content_recommender = ContentBasedRecommender(self.data)
        self.collaborative_recommender = CollaborativeRecommender(self.data)
        
        # Load models if available
        try:
            self.content_recommender.load_models()
            print("Content-based models loaded successfully.")
        except Exception as e:
            print(f"Could not load content-based models: {e}")
        
        try:
            self.collaborative_recommender.load_models()
            print("Collaborative filtering models loaded successfully.")
        except Exception as e:
            print(f"Could not load collaborative filtering models: {e}")
    
    def get_weighted_recommendations(self, user_id=None, paper_title=None, content_weight=0.5, top_n=10):
        """Get recommendations using a weighted combination of content and collaborative filtering."""
        if user_id is None and paper_title is None:
            raise ValueError("Either user_id or paper_title must be provided.")
        
        # Get content-based recommendations
        if paper_title is not None:
            content_recs = self.content_recommender.get_recommendations_by_title(paper_title, method='semantic', top_n=top_n*2)
            content_paper_ids = content_recs['id'].tolist()
            content_scores = content_recs['similarity_score'].tolist() if 'similarity_score' in content_recs.columns else [1.0] * len(content_paper_ids)
        else:
            # If no paper title, use a random paper from the user's history
            if self.collaborative_recommender.user_data is not None:
                user_papers = self.collaborative_recommender.user_data[
                    self.collaborative_recommender.user_data['user_id'] == user_id
                ]
                if not user_papers.empty:
                    random_paper_id = user_papers.sample(1)['paper_id'].iloc[0]
                    paper_idx = self.data[self.data['id'] == random_paper_id].index[0]
                    content_recs = self.content_recommender.get_semantic_recommendations(paper_idx, top_n=top_n*2)
                    content_paper_ids = content_recs['id'].tolist()
                    content_scores = content_recs['similarity_score'].tolist() if 'similarity_score' in content_recs.columns else [1.0] * len(content_paper_ids)
                else:
                    content_paper_ids = []
                    content_scores = []
            else:
                content_paper_ids = []
                content_scores = []
        
        # Get collaborative filtering recommendations
        if user_id is not None and self.collaborative_recommender.svd_model is not None:
            collab_recs = self.collaborative_recommender.get_top_n_recommendations(user_id, n=top_n*2, model='svd')
            collab_paper_ids = collab_recs['id'].tolist()
            collab_scores = collab_recs['similarity_score'].tolist() if 'similarity_score' in collab_recs.columns else [1.0] * len(collab_paper_ids)
        else:
            collab_paper_ids = []
            collab_scores = []
        
        # Combine the recommendations
        if not content_paper_ids and not collab_paper_ids:
            return pd.DataFrame()
        
        # Create a dictionary to store the weighted scores
        paper_scores = {}
        
        # Add content-based scores
        for i, (paper_id, score) in enumerate(zip(content_paper_ids, content_scores)):
            # Normalize the score if needed
            normalized_score = score
            paper_scores[paper_id] = content_weight * normalized_score
        
        # Add collaborative filtering scores
        for i, (paper_id, score) in enumerate(zip(collab_paper_ids, collab_scores)):
            # Normalize the score if needed
            normalized_score = score
            if paper_id in paper_scores:
                paper_scores[paper_id] += (1 - content_weight) * normalized_score
            else:
                paper_scores[paper_id] = (1 - content_weight) * normalized_score
        
        # Sort papers by score
        sorted_papers = sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get the top N paper IDs and scores
        top_paper_ids = [p[0] for p in sorted_papers[:top_n]]
        top_paper_scores = [p[1] for p in sorted_papers[:top_n]]
        
        # Return the corresponding papers with similarity scores
        result_df = self.data[self.data['id'].isin(top_paper_ids)].copy()
        
        # Add similarity scores to the dataframe
        score_dict = dict(zip(top_paper_ids, top_paper_scores))
        result_df['similarity_score'] = result_df['id'].map(score_dict)
        
        # Sort by similarity score
        result_df = result_df.sort_values('similarity_score', ascending=False)
        
        return result_df
    
    def get_recommendations_by_categories_and_user(self, user_id, categories, content_weight=0.7, top_n=10):
        """Get recommendations based on categories and user preferences."""
        # Get content-based recommendations by categories
        content_recs = self.content_recommender.get_recommendations_by_categories(categories, top_n=top_n*2)
        content_paper_ids = content_recs['id'].tolist() if not content_recs.empty else []
        content_scores = content_recs['similarity_score'].tolist() if not content_recs.empty and 'similarity_score' in content_recs.columns else [1.0] * len(content_paper_ids)
        
        # Get collaborative filtering recommendations
        if self.collaborative_recommender.svd_model is not None:
            collab_recs = self.collaborative_recommender.get_top_n_recommendations(user_id, n=top_n*2, model='svd')
            collab_paper_ids = collab_recs['id'].tolist()
            collab_scores = collab_recs['similarity_score'].tolist() if 'similarity_score' in collab_recs.columns else [1.0] * len(collab_paper_ids)
        else:
            collab_paper_ids = []
            collab_scores = []
        
        # Combine the recommendations
        if not content_paper_ids and not collab_paper_ids:
            return pd.DataFrame()
        
        # Create a dictionary to store the weighted scores
        paper_scores = {}
        
        # Add content-based scores
        for i, (paper_id, score) in enumerate(zip(content_paper_ids, content_scores)):
            # Normalize the score if needed
            normalized_score = score
            paper_scores[paper_id] = content_weight * normalized_score
        
        # Add collaborative filtering scores
        for i, (paper_id, score) in enumerate(zip(collab_paper_ids, collab_scores)):
            # Normalize the score if needed
            normalized_score = score
            if paper_id in paper_scores:
                paper_scores[paper_id] += (1 - content_weight) * normalized_score
            else:
                paper_scores[paper_id] = (1 - content_weight) * normalized_score
        
        # Sort papers by score
        sorted_papers = sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get the top N paper IDs and scores
        top_paper_ids = [p[0] for p in sorted_papers[:top_n]]
        top_paper_scores = [p[1] for p in sorted_papers[:top_n]]
        
        # Return the corresponding papers with similarity scores
        result_df = self.data[self.data['id'].isin(top_paper_ids)].copy()
        
        # Add similarity scores to the dataframe
        score_dict = dict(zip(top_paper_ids, top_paper_scores))
        result_df['similarity_score'] = result_df['id'].map(score_dict)
        
        # Sort by similarity score
        result_df = result_df.sort_values('similarity_score', ascending=False)
        
        return result_df
    
    def get_recommendations_by_authors_and_user(self, user_id, authors, content_weight=0.7, top_n=10):
        """Get recommendations based on authors and user preferences."""
        # Get content-based recommendations by authors
        content_recs = self.content_recommender.get_recommendations_by_authors(authors, top_n=top_n*2)
        content_paper_ids = content_recs['id'].tolist() if not content_recs.empty else []
        content_scores = content_recs['similarity_score'].tolist() if not content_recs.empty and 'similarity_score' in content_recs.columns else [1.0] * len(content_paper_ids)
        
        # Get collaborative filtering recommendations
        if self.collaborative_recommender.svd_model is not None:
            collab_recs = self.collaborative_recommender.get_top_n_recommendations(user_id, n=top_n*2, model='svd')
            collab_paper_ids = collab_recs['id'].tolist()
            collab_scores = collab_recs['similarity_score'].tolist() if 'similarity_score' in collab_recs.columns else [1.0] * len(collab_paper_ids)
        else:
            collab_paper_ids = []
            collab_scores = []
        
        # Combine the recommendations
        if not content_paper_ids and not collab_paper_ids:
            return pd.DataFrame()
        
        # Create a dictionary to store the weighted scores
        paper_scores = {}
        
        # Add content-based scores
        for i, (paper_id, score) in enumerate(zip(content_paper_ids, content_scores)):
            # Normalize the score if needed
            normalized_score = score
            paper_scores[paper_id] = content_weight * normalized_score
        
        # Add collaborative filtering scores
        for i, (paper_id, score) in enumerate(zip(collab_paper_ids, collab_scores)):
            # Normalize the score if needed
            normalized_score = score
            if paper_id in paper_scores:
                paper_scores[paper_id] += (1 - content_weight) * normalized_score
            else:
                paper_scores[paper_id] = (1 - content_weight) * normalized_score
        
        # Sort papers by score
        sorted_papers = sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get the top N paper IDs and scores
        top_paper_ids = [p[0] for p in sorted_papers[:top_n]]
        top_paper_scores = [p[1] for p in sorted_papers[:top_n]]
        
        # Return the corresponding papers with similarity scores
        result_df = self.data[self.data['id'].isin(top_paper_ids)].copy()
        
        # Add similarity scores to the dataframe
        score_dict = dict(zip(top_paper_ids, top_paper_scores))
        result_df['similarity_score'] = result_df['id'].map(score_dict)
        
        # Sort by similarity score
        result_df = result_df.sort_values('similarity_score', ascending=False)
        
        return result_df
    
    def train_models(self):
        """Train all the component models."""
        # Train content-based models
        self.content_recommender.build_tfidf_model()
        self.content_recommender.build_transformer_model()
        self.content_recommender.build_faiss_index()
        
        # Train collaborative filtering models
        self.collaborative_recommender.simulate_user_data()
        self.collaborative_recommender.prepare_surprise_data()
        self.collaborative_recommender.train_svd_model()
        self.collaborative_recommender.train_svdpp_model()
    
    def save_models(self, output_dir='.'):
        """Save all the trained models to files."""
        self.content_recommender.save_models(output_dir)
        self.collaborative_recommender.save_models(output_dir)

if __name__ == "__main__":
    # Example usage
    recommender = HybridRecommender()
    recommender.train_models()
    recommender.save_models()
    print("Hybrid recommender models built and saved.")
