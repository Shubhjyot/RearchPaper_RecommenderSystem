import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import joblib
from sentence_transformers import SentenceTransformer
import time
import os

class ContentBasedRecommender:
    def __init__(self, processed_data=None, data_path='processed_data.pkl'):
        """Initialize the content-based recommender with processed data."""
        if processed_data is not None:
            self.data = processed_data
        elif os.path.exists(data_path):
            self.data = joblib.load(data_path)
        else:
            raise FileNotFoundError(f"Processed data file {data_path} not found.")
        
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.faiss_index = None
        self.transformer_model = None
        self.embeddings = None
    
    def build_tfidf_model(self):
        """Build TF-IDF model for text-based similarity."""
        print("Building TF-IDF model...")
        start_time = time.time()
        
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.data['combined_features'])
        
        print(f"TF-IDF model built in {time.time() - start_time:.2f} seconds.")
        return self.tfidf_matrix
    
    def build_transformer_model(self, model_name='all-MiniLM-L6-v2'):
        """Build transformer model for semantic similarity."""
        print(f"Building transformer model with {model_name}...")
        start_time = time.time()
        
        self.transformer_model = SentenceTransformer(model_name)
        
        # Combine title and summary for better semantic understanding
        text_for_embedding = self.data['title'] + ". " + self.data['summary']
        
        # Generate embeddings
        self.embeddings = self.transformer_model.encode(
            text_for_embedding.tolist(), 
            show_progress_bar=True,
            batch_size=32
        )
        
        print(f"Transformer model built in {time.time() - start_time:.2f} seconds.")
        return self.embeddings
    
    def build_faiss_index(self):
        """Build FAISS index for fast similarity search."""
        if self.embeddings is None:
            raise ValueError("Embeddings not available. Run build_transformer_model first.")
        
        print("Building FAISS index...")
        start_time = time.time()
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = self.embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings_normalized)
        
        # Create FAISS index
        dimension = embeddings_normalized.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity with normalized vectors
        self.faiss_index.add(embeddings_normalized)
        
        print(f"FAISS index built in {time.time() - start_time:.2f} seconds.")
        return self.faiss_index
    
    def get_tfidf_recommendations(self, paper_idx, top_n=10):
        """Get recommendations based on TF-IDF similarity."""
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF matrix not available. Run build_tfidf_model first.")
        
        # Get the similarity scores for the paper
        paper_vector = self.tfidf_matrix[paper_idx]
        similarity_scores = cosine_similarity(paper_vector, self.tfidf_matrix).flatten()
        
        # Get the indices of the top similar papers (excluding the paper itself)
        similar_indices = similarity_scores.argsort()[::-1]
        similar_indices = similar_indices[similar_indices != paper_idx][:top_n]
        similar_scores = similarity_scores[similar_indices]
        
        # Return the similar papers with similarity scores
        result_df = self.data.iloc[similar_indices].copy()
        result_df['similarity_score'] = similar_scores
        
        return result_df
    
    def get_semantic_recommendations(self, paper_idx, top_n=10):
        """Get recommendations based on semantic similarity using FAISS."""
        if self.faiss_index is None:
            raise ValueError("FAISS index not available. Run build_faiss_index first.")
        
        # Get the embedding for the paper
        query_embedding = self.embeddings[paper_idx].reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search for similar papers
        distances, indices = self.faiss_index.search(query_embedding, top_n + 1)
        
        # Filter out the query paper itself
        mask = indices[0] != paper_idx
        filtered_indices = indices[0][mask][:top_n]
        filtered_distances = distances[0][mask][:top_n]
        
        # Convert distances to similarity scores (1 - normalized distance)
        # FAISS returns inner product which is already a similarity measure for normalized vectors
        similarity_scores = filtered_distances
        
        # Return the similar papers with similarity scores
        result_df = self.data.iloc[filtered_indices].copy()
        result_df['similarity_score'] = similarity_scores
        
        return result_df
    
    def get_recommendations_by_title(self, title, method='semantic', top_n=10):
        """Get recommendations based on paper title."""
        # Find the paper with the closest matching title
        title_lower = title.lower()
        self.data['title_lower'] = self.data['title'].str.lower()
        
        # Try exact match first
        exact_matches = self.data[self.data['title_lower'] == title_lower]
        
        if not exact_matches.empty:
            paper_idx = exact_matches.index[0]
        else:
            # If no exact match, find the closest match
            self.data['title_similarity'] = self.data['title_lower'].apply(
                lambda x: len(set(x.split()) & set(title_lower.split())) / len(set(x.split()) | set(title_lower.split()))
                if isinstance(x, str) else 0
            )
            paper_idx = self.data['title_similarity'].argmax()
        
        # Get recommendations based on the specified method
        if method == 'tfidf':
            return self.get_tfidf_recommendations(paper_idx, top_n)
        elif method == 'semantic':
            return self.get_semantic_recommendations(paper_idx, top_n)
        else:
            raise ValueError("Method must be either 'tfidf' or 'semantic'")
    
    def get_recommendations_by_categories(self, categories, top_n=10):
        """Get recommendations based on categories."""
        # Convert categories to lowercase
        categories_lower = [cat.lower() for cat in categories]
        
        # Filter papers that have at least one matching category
        matching_papers = self.data[
            self.data['categories'].apply(
                lambda x: any(cat in x.lower() for cat in categories_lower) if isinstance(x, str) else False
            )
        ].copy()  # Create a copy to avoid SettingWithCopyWarning
        
        if matching_papers.empty:
            return pd.DataFrame()
        
        # Count the number of matching categories for each paper
        matching_papers.loc[:, 'category_matches'] = matching_papers['categories'].apply(
            lambda x: sum(cat in x.lower() for cat in categories_lower) if isinstance(x, str) else 0
        )
        
        # Calculate similarity score based on category matches
        total_categories = len(categories_lower)
        matching_papers.loc[:, 'similarity_score'] = matching_papers['category_matches'] / total_categories
        
        # Sort by the number of matching categories and return top_n
        return matching_papers.sort_values('similarity_score', ascending=False).head(top_n)
    
    def get_recommendations_by_authors(self, authors, top_n=10):
        """Get recommendations based on authors."""
        # Convert authors to lowercase
        authors_lower = [author.lower() for author in authors]
        
        # Filter papers that have at least one matching author
        matching_papers = self.data[
            self.data['authors'].apply(
                lambda x: any(author in x.lower() for author in authors_lower) if isinstance(x, str) else False
            )
        ].copy()  # Create a copy to avoid SettingWithCopyWarning
        
        if matching_papers.empty:
            return pd.DataFrame()
        
        # Count the number of matching authors for each paper
        matching_papers.loc[:, 'author_matches'] = matching_papers['authors'].apply(
            lambda x: sum(author in x.lower() for author in authors_lower) if isinstance(x, str) else 0
        )
        
        # Calculate similarity score based on author matches
        total_authors = len(authors_lower)
        matching_papers.loc[:, 'similarity_score'] = matching_papers['author_matches'] / total_authors
        
        # Sort by the number of matching authors and return top_n
        return matching_papers.sort_values('similarity_score', ascending=False).head(top_n)
    
    def save_models(self, output_dir='.'):
        """Save the trained models to files."""
        if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
            joblib.dump(self.tfidf_vectorizer, f"{output_dir}/tfidf_vectorizer.pkl")
            joblib.dump(self.tfidf_matrix, f"{output_dir}/tfidf_matrix.pkl")
            print(f"TF-IDF model saved to {output_dir}")
        
        if self.transformer_model is not None and self.embeddings is not None:
            # Save embeddings
            np.save(f"{output_dir}/embeddings.npy", self.embeddings)
            print(f"Embeddings saved to {output_dir}")
        
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, f"{output_dir}/faiss_index.bin")
            print(f"FAISS index saved to {output_dir}")
    
    def load_models(self, input_dir='.'):
        """Load the trained models from files."""
        # Load TF-IDF model
        if os.path.exists(f"{input_dir}/tfidf_vectorizer.pkl") and os.path.exists(f"{input_dir}/tfidf_matrix.pkl"):
            self.tfidf_vectorizer = joblib.load(f"{input_dir}/tfidf_vectorizer.pkl")
            self.tfidf_matrix = joblib.load(f"{input_dir}/tfidf_matrix.pkl")
            print(f"TF-IDF model loaded from {input_dir}")
        
        # Load embeddings
        if os.path.exists(f"{input_dir}/embeddings.npy"):
            self.embeddings = np.load(f"{input_dir}/embeddings.npy")
            print(f"Embeddings loaded from {input_dir}")
        
        # Load FAISS index
        if os.path.exists(f"{input_dir}/faiss_index.bin"):
            self.faiss_index = faiss.read_index(f"{input_dir}/faiss_index.bin")
            print(f"FAISS index loaded from {input_dir}")

if __name__ == "__main__":
    # Example usage
    recommender = ContentBasedRecommender()
    recommender.build_tfidf_model()
    recommender.build_transformer_model()
    recommender.build_faiss_index()
    recommender.save_models()
    print("Content-based models built and saved.")
