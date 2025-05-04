import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
from tqdm import tqdm
import joblib
import os

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class DataPreprocessor:
    def __init__(self, data_path):
        """Initialize the data preprocessor with the path to the data file."""
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
    def load_data(self):
        """Load the data from the CSV file."""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} papers.")
        return self.df
    
    def clean_text(self, text):
        """Clean and preprocess text data."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        filtered_tokens = [self.stemmer.stem(w) for w in tokens if w not in self.stop_words]
        
        return ' '.join(filtered_tokens)
    
    def preprocess_data(self):
        """Preprocess the data for the recommender system."""
        if self.df is None:
            self.load_data()
        
        print("Preprocessing data...")
        
        # Create a copy of the dataframe for processing
        self.processed_df = self.df.copy()
        
        # Clean text fields
        tqdm.pandas(desc="Processing titles")
        self.processed_df['processed_title'] = self.processed_df['title'].progress_apply(self.clean_text)
        
        tqdm.pandas(desc="Processing summaries")
        self.processed_df['processed_summary'] = self.processed_df['summary'].progress_apply(self.clean_text)
        
        # Process authors
        tqdm.pandas(desc="Processing authors")
        self.processed_df['processed_authors'] = self.processed_df['authors'].progress_apply(
            lambda x: ' '.join(x.lower().split(',')) if isinstance(x, str) else ""
        )
        
        # Process categories
        tqdm.pandas(desc="Processing categories")
        self.processed_df['processed_categories'] = self.processed_df['categories'].progress_apply(
            lambda x: ' '.join(x.lower().split()) if isinstance(x, str) else ""
        )
        
        # Create a combined text field for content-based filtering
        self.processed_df['combined_features'] = (
            self.processed_df['processed_title'] + ' ' +
            self.processed_df['processed_summary'] + ' ' +
            self.processed_df['processed_authors'] + ' ' +
            self.processed_df['processed_categories']
        )
        
        print("Data preprocessing completed.")
        return self.processed_df
    
    def save_processed_data(self, output_path='processed_data.pkl'):
        """Save the processed data to a file."""
        if self.processed_df is None:
            raise ValueError("No processed data available. Run preprocess_data first.")
        
        joblib.dump(self.processed_df, output_path)
        print(f"Processed data saved to {output_path}")
    
    def load_processed_data(self, input_path='processed_data.pkl'):
        """Load preprocessed data from a file."""
        if os.path.exists(input_path):
            self.processed_df = joblib.load(input_path)
            print(f"Loaded processed data from {input_path}")
            return self.processed_df
        else:
            raise FileNotFoundError(f"Processed data file {input_path} not found.")

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor("arxiv_papers_unique_20250425_Final.csv")
    processed_data = preprocessor.preprocess_data()
    preprocessor.save_processed_data()
    print("Data preprocessing completed and saved.")
