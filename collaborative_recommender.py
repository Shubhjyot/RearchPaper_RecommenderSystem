import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, SVDpp
from surprise.model_selection import train_test_split
import joblib
import os
import time
from collections import defaultdict

class CollaborativeRecommender:
    def __init__(self, processed_data=None, data_path='processed_data.pkl'):
        """Initialize the collaborative recommender with processed data."""
        if processed_data is not None:
            self.data = processed_data
        elif os.path.exists(data_path):
            self.data = joblib.load(data_path)
        else:
            raise FileNotFoundError(f"Processed data file {data_path} not found.")
        
        self.user_data = None
        self.svd_model = None
        self.svdpp_model = None
        self.reader = None
        self.trainset = None
        self.testset = None
    
    def simulate_user_data(self, n_users=1000, rating_sparsity=0.01, seed=42):
        """Simulate user interactions with papers."""
        print(f"Simulating data for {n_users} users...")
        start_time = time.time()
        
        np.random.seed(seed)
        
        # Get the number of papers
        n_papers = len(self.data)
        
        # Create user IDs
        user_ids = [f"user_{i}" for i in range(n_users)]
        
        # Determine how many ratings each user will give
        ratings_per_user = int(n_papers * rating_sparsity)
        
        # Generate ratings
        ratings = []
        
        for user_id in user_ids:
            # Select random papers for this user
            paper_indices = np.random.choice(
                n_papers, 
                size=ratings_per_user, 
                replace=False
            )
            
            # Generate ratings (1-5) with a bias towards higher ratings for more cited papers
            for idx in paper_indices:
                # Simulate rating based on paper popularity or quality
                # Here we're using a simple random distribution
                rating = np.random.choice(
                    [3, 3.5, 4, 4.5, 5], 
                    p=[0.1, 0.2, 0.3, 0.25, 0.15]
                )
                
                ratings.append({
                    'user_id': user_id,
                    'paper_id': self.data.iloc[idx]['id'],
                    'rating': rating
                })
        
        # Create a DataFrame
        self.user_data = pd.DataFrame(ratings)
        
        print(f"Generated {len(self.user_data)} ratings in {time.time() - start_time:.2f} seconds.")
        return self.user_data
    
    def prepare_surprise_data(self):
        """Prepare data for Surprise library."""
        if self.user_data is None:
            raise ValueError("User data not available. Run simulate_user_data first.")
        
        print("Preparing data for Surprise...")
        
        # Define the reader
        self.reader = Reader(rating_scale=(1, 5))
        
        # Load the data
        data = Dataset.load_from_df(
            self.user_data[['user_id', 'paper_id', 'rating']], 
            self.reader
        )
        
        # Split the data
        self.trainset, self.testset = train_test_split(data, test_size=0.2)
        
        print("Data preparation completed.")
        return self.trainset, self.testset
    
    def train_svd_model(self, n_factors=100, n_epochs=20):
        """Train an SVD model for collaborative filtering."""
        if self.trainset is None:
            raise ValueError("Training data not available. Run prepare_surprise_data first.")
        
        print(f"Training SVD model with {n_factors} factors...")
        start_time = time.time()
        
        # Initialize and train the model
        self.svd_model = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=42)
        self.svd_model.fit(self.trainset)
        
        print(f"SVD model trained in {time.time() - start_time:.2f} seconds.")
        return self.svd_model
    
    def train_svdpp_model(self, n_factors=100, n_epochs=20):
        """Train an SVD++ model for collaborative filtering."""
        if self.trainset is None:
            raise ValueError("Training data not available. Run prepare_surprise_data first.")
        
        print(f"Training SVD++ model with {n_factors} factors...")
        start_time = time.time()
        
        # Initialize and train the model
        self.svdpp_model = SVDpp(n_factors=n_factors, n_epochs=n_epochs, random_state=42)
        self.svdpp_model.fit(self.trainset)
        
        print(f"SVD++ model trained in {time.time() - start_time:.2f} seconds.")
        return self.svdpp_model
    
    def get_top_n_recommendations(self, user_id, n=10, model='svd'):
        """Get top N recommendations for a user."""
        if model == 'svd' and self.svd_model is None:
            raise ValueError("SVD model not available. Run train_svd_model first.")
        elif model == 'svdpp' and self.svdpp_model is None:
            raise ValueError("SVD++ model not available. Run train_svdpp_model first.")
        
        # Get the model to use
        selected_model = self.svd_model if model == 'svd' else self.svdpp_model
        
        # Get all paper IDs
        all_paper_ids = self.data['id'].tolist()
        
        # Get papers the user has already rated
        if self.user_data is not None:
            rated_paper_ids = self.user_data[self.user_data['user_id'] == user_id]['paper_id'].tolist()
        else:
            rated_paper_ids = []
        
        # Get papers the user hasn't rated
        unrated_paper_ids = [pid for pid in all_paper_ids if pid not in rated_paper_ids]
        
        # Predict ratings for unrated papers
        predictions = []
        for paper_id in unrated_paper_ids:
            predicted_rating = selected_model.predict(user_id, paper_id).est
            predictions.append((paper_id, predicted_rating))
        
        # Sort predictions by rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get the top N paper IDs and their scores
        top_paper_ids = [p[0] for p in predictions[:n]]
        top_paper_scores = [p[1] for p in predictions[:n]]
        
        # Return the corresponding papers with similarity scores
        result_df = self.data[self.data['id'].isin(top_paper_ids)].copy()
        
        # Add similarity scores to the dataframe
        score_dict = dict(zip(top_paper_ids, top_paper_scores))
        result_df['similarity_score'] = result_df['id'].map(score_dict)
        
        # Sort by similarity score
        result_df = result_df.sort_values('similarity_score', ascending=False)
        
        return result_df
    
    def get_recommendations_for_new_user(self, paper_ids, ratings, n=10, model='svd'):
        """Get recommendations for a new user based on their ratings."""
        if model == 'svd' and self.svd_model is None:
            raise ValueError("SVD model not available. Run train_svd_model first.")
        elif model == 'svdpp' and self.svdpp_model is None:
            raise ValueError("SVD++ model not available. Run train_svdpp_model first.")
        
        # Get the model to use
        selected_model = self.svd_model if model == 'svd' else self.svdpp_model
        
        # Create a temporary user ID
        temp_user_id = "temp_user"
        
        # Get all paper IDs
        all_paper_ids = self.data['id'].tolist()
        
        # Get papers the user hasn't rated
        unrated_paper_ids = [pid for pid in all_paper_ids if pid not in paper_ids]
        
        # Add the user's ratings to the model using partial_fit
        for paper_id, rating in zip(paper_ids, ratings):
            selected_model.trainset.raw_ratings.append((temp_user_id, paper_id, rating, None))
        
        # Rebuild the model's user factors for the new user
        selected_model.partial_fit([(temp_user_id, paper_id, rating) for paper_id, rating in zip(paper_ids, ratings)])
        
        # Predict ratings for unrated papers
        predictions = []
        for paper_id in unrated_paper_ids:
            predicted_rating = selected_model.predict(temp_user_id, paper_id).est
            predictions.append((paper_id, predicted_rating))
        
        # Sort predictions by rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get the top N paper IDs and their scores
        top_paper_ids = [p[0] for p in predictions[:n]]
        top_paper_scores = [p[1] for p in predictions[:n]]
        
        # Return the corresponding papers with similarity scores
        result_df = self.data[self.data['id'].isin(top_paper_ids)].copy()
        
        # Add similarity scores to the dataframe
        score_dict = dict(zip(top_paper_ids, top_paper_scores))
        result_df['similarity_score'] = result_df['id'].map(score_dict)
        
        # Sort by similarity score
        result_df = result_df.sort_values('similarity_score', ascending=False)
        
        return result_df
    
    def save_models(self, output_dir='.'):
        """Save the trained models and user data to files."""
        if self.user_data is not None:
            self.user_data.to_csv(f"{output_dir}/simulated_user_data.csv", index=False)
            print(f"Simulated user data saved to {output_dir}")
        
        if self.svd_model is not None:
            joblib.dump(self.svd_model, f"{output_dir}/svd_model.pkl")
            print(f"SVD model saved to {output_dir}")
        
        if self.svdpp_model is not None:
            joblib.dump(self.svdpp_model, f"{output_dir}/svdpp_model.pkl")
            print(f"SVD++ model saved to {output_dir}")
    
    def load_models(self, input_dir='.'):
        """Load the trained models and user data from files."""
        # Load user data
        if os.path.exists(f"{input_dir}/simulated_user_data.csv"):
            self.user_data = pd.read_csv(f"{input_dir}/simulated_user_data.csv")
            print(f"Simulated user data loaded from {input_dir}")
        
        # Load SVD model
        if os.path.exists(f"{input_dir}/svd_model.pkl"):
            self.svd_model = joblib.load(f"{input_dir}/svd_model.pkl")
            print(f"SVD model loaded from {input_dir}")
        
        # Load SVD++ model
        if os.path.exists(f"{input_dir}/svdpp_model.pkl"):
            self.svdpp_model = joblib.load(f"{input_dir}/svdpp_model.pkl")
            print(f"SVD++ model loaded from {input_dir}")

if __name__ == "__main__":
    # Example usage
    recommender = CollaborativeRecommender()
    recommender.simulate_user_data()
    recommender.prepare_surprise_data()
    recommender.train_svd_model()
    recommender.train_svdpp_model()
    recommender.save_models()
    print("Collaborative filtering models built and saved.")
