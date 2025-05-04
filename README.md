# Paper Recommender System

A comprehensive academic paper recommender system with multiple recommendation approaches and a conversational interface.

## Features

- **Content-Based Filtering**: Recommends papers based on title, authors, and categories similarity
- **Collaborative Filtering**: Uses SVD and SVD++ algorithms to recommend papers based on user preferences
- **Hybrid Recommendations**: Combines content-based and collaborative approaches for better suggestions
- **Conversational Interface**: Chat with an AI assistant to get paper recommendations using RAG and Gemini
- **Vector Search**: Fast similarity search using FAISS
- **Streamlit UI**: User-friendly interface with caching for improved performance

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. For the conversational interface, you need a Google Gemini API key:
   - Copy `.env.example` to `.env`
   - Add your Gemini API key to the `.env` file
   - Or enter it directly in the app when prompted

## Running the Application

Start the Streamlit app:

```
streamlit run app.py
```

## Components

- `data_preprocessing.py`: Loads and preprocesses the arXiv papers dataset
- `content_based_recommender.py`: Implements content-based filtering using TF-IDF and transformer models
- `collaborative_recommender.py`: Implements collaborative filtering with SVD and SVD++ algorithms
- `hybrid_recommender.py`: Combines both recommendation approaches
- `conversational_rag.py`: Implements a conversational interface with RAG using Gemini
- `app.py`: Streamlit application that ties everything together

## Usage

1. **Load Data & Models**: Click the button in the sidebar to load the dataset and train/load models
2. **Content-Based Recommendations**: Search for papers by title, authors, or categories
3. **Collaborative Recommendations**: Get personalized recommendations based on your ratings
4. **Hybrid Recommendations**: Combine content-based and collaborative approaches
5. **Chat with Assistant**: Ask questions and get paper recommendations through natural language

## Data

The system uses the arXiv papers dataset with the following fields:
- Title
- Summary
- Authors
- Categories
- PDF URL
- Publication date

## Performance Optimizations

- FAISS vector database for fast similarity search
- Caching of model results for improved performance
- Efficient data preprocessing with NLTK
- Streamlit caching for UI responsiveness
