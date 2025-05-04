import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import plotly.express as px
from data_preprocessing import DataPreprocessor
from content_based_recommender import ContentBasedRecommender
from collaborative_recommender import CollaborativeRecommender
from hybrid_recommender import HybridRecommender
from conversational_rag import ConversationalRAG
import joblib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Paper Recommender System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .paper-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 5px solid #4CAF50;
    }
    .paper-title {
        font-weight: bold;
        color: #2C3E50;
    }
    .paper-authors {
        font-style: italic;
        color: #7F8C8D;
    }
    .paper-categories {
        color: #3498DB;
    }
    .paper-summary {
        color: #34495E;
    }
    .paper-url {
        color: #E74C3C;
    }
    .paper-score {
        font-size: 12px;
        color: #666;
    }
    .chat-user {
        background-color: #E8F4F8;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .chat-assistant {
        background-color: #F0F7F4;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = f"user_{np.random.randint(1000, 9999)}"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'papers_df' not in st.session_state:
    st.session_state.papers_df = None
if 'content_recommender' not in st.session_state:
    st.session_state.content_recommender = None
if 'collaborative_recommender' not in st.session_state:
    st.session_state.collaborative_recommender = None
if 'hybrid_recommender' not in st.session_state:
    st.session_state.hybrid_recommender = None
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

# Function to load data and models
@st.cache_resource
def load_data_and_models():
    """Load data and models with caching."""
    # Check if processed data exists
    if os.path.exists('processed_data.pkl'):
        papers_df = joblib.load('processed_data.pkl')
    else:
        # Load and preprocess data
        preprocessor = DataPreprocessor("arxiv_papers_unique_20250425_Final.csv")
        papers_df = preprocessor.preprocess_data()
        preprocessor.save_processed_data()
    
    # Initialize recommenders
    content_recommender = ContentBasedRecommender(papers_df)
    collaborative_recommender = CollaborativeRecommender(papers_df)
    hybrid_recommender = HybridRecommender(papers_df)
    rag_system = ConversationalRAG(papers_df)
    
    # Try to load pre-trained models
    try:
        content_recommender.load_models()
    except Exception as e:
        st.warning(f"Could not load content-based models: {e}")
        st.info("Building content-based models...")
        content_recommender.build_tfidf_model()
        content_recommender.build_transformer_model()
        content_recommender.build_faiss_index()
        content_recommender.save_models()
    
    try:
        collaborative_recommender.load_models()
    except Exception as e:
        st.warning(f"Could not load collaborative filtering models: {e}")
        st.info("Building collaborative filtering models...")
        collaborative_recommender.simulate_user_data()
        collaborative_recommender.prepare_surprise_data()
        collaborative_recommender.train_svd_model()
        collaborative_recommender.train_svdpp_model()
        collaborative_recommender.save_models()
    
    try:
        rag_system.load_models()
    except Exception as e:
        st.warning(f"Could not load RAG models: {e}")
        st.info("Building RAG models...")
        rag_system.build_vector_db()
        rag_system.save_models()
    
    return papers_df, content_recommender, collaborative_recommender, hybrid_recommender, rag_system

# Function to display paper cards
def display_paper_card(paper):
    """Display a paper in a card format."""
    # Check if similarity score is available
    similarity_score = ""
    if 'similarity_score' in paper and not pd.isna(paper['similarity_score']):
        score = paper['similarity_score']
        # Format the score as percentage if it's between 0 and 1
        if 0 <= score <= 1:
            similarity_score = f"<div class='paper-score'>Similarity Score: {score:.2f} ({score*100:.1f}%)</div>"
        else:
            similarity_score = f"<div class='paper-score'>Similarity Score: {score:.2f}</div>"
    
    with st.container():
        st.markdown(f"""
        <div class="paper-card">
            <div class="paper-title">{paper['title']}</div>
            {similarity_score}
            <div class="paper-authors">Authors: {paper['authors']}</div>
            <div class="paper-categories">Categories: {paper['categories']}</div>
            <div class="paper-summary">Summary: {paper['summary'][:300]}...</div>
            <div class="paper-url">URL: <a href="{paper['pdf_url']}" target="_blank">{paper['pdf_url']}</a></div>
        </div>
        """, unsafe_allow_html=True)

# Function to handle chat input
def handle_chat_input():
    """Handle chat input and generate responses."""
    if st.session_state.chat_input and st.session_state.rag_system:
        user_input = st.session_state.chat_input
        
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Generate response
        with st.spinner("Thinking..."):
            response = st.session_state.rag_system.generate_response(user_input)
        
        # Add assistant message to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Clear input
        st.session_state.chat_input = ""

# Main app
def main():
    # Sidebar
    with st.sidebar:
        st.title("📚 Paper Recommender")
        st.write(f"User ID: {st.session_state.user_id}")
        
        # Load data and models button
        if not st.session_state.data_loaded:
            if st.button("Load Data & Models"):
                with st.spinner("Loading data and models..."):
                    (
                        st.session_state.papers_df,
                        st.session_state.content_recommender,
                        st.session_state.collaborative_recommender,
                        st.session_state.hybrid_recommender,
                        st.session_state.rag_system
                    ) = load_data_and_models()
                    st.session_state.data_loaded = True
                    st.session_state.models_loaded = True
                st.success("Data and models loaded successfully!")
                st.rerun()
        else:
            st.success("Data and models loaded!")
        
        # Navigation
        st.header("Navigation")
        page = st.radio(
            "Select a page",
            ["Home", "Content-Based Recommendations", "Collaborative Recommendations", 
             "Hybrid Recommendations", "Chat with RAG Assistant"]
        )
    
    # Main content
    if not st.session_state.data_loaded:
        st.title("📚 Welcome to the Paper Recommender System")
        st.write("""
        This application helps you discover academic papers based on your interests and preferences.
        It uses multiple recommendation techniques:
        
        - **Content-Based Filtering**: Recommends papers similar to ones you like
        - **Collaborative Filtering**: Recommends papers that similar users liked
        - **Hybrid Approach**: Combines both methods for better recommendations
        - **Conversational Assistant**: Chat to get personalized paper recommendations
        
        To get started, click the "Load Data & Models" button in the sidebar.
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("📊 63,000+ Papers")
        with col2:
            st.info("🔍 Fast Search with FAISS")
        with col3:
            st.info("🤖 AI-Powered Recommendations")
        
        return
    
    # Home page
    if page == "Home":
        st.title("📚 Paper Recommender System")
        
        # Paper statistics
        st.header("Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Papers", f"{len(st.session_state.papers_df):,}")
        
        # Most common categories
        categories = []
        for cats in st.session_state.papers_df['categories'].dropna():
            categories.extend([c.strip() for c in cats.split()])
        
        category_counts = pd.Series(categories).value_counts().head(10)
        
        with col2:
            st.metric("Unique Categories", f"{len(pd.Series(categories).value_counts()):,}")
        
        with col3:
            st.metric("Average Summary Length", f"{int(st.session_state.papers_df['summary'].str.len().mean()):,} chars")
        
        # Category distribution
        st.subheader("Top 10 Categories")
        fig = px.bar(
            category_counts, 
            x=category_counts.index, 
            y=category_counts.values,
            labels={'x': 'Category', 'y': 'Count'},
            color=category_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample papers
        st.header("Sample Papers")
        sample_papers = st.session_state.papers_df.sample(5)
        for _, paper in sample_papers.iterrows():
            display_paper_card(paper)
    
    # Content-Based Recommendations
    elif page == "Content-Based Recommendations":
        st.title("Content-Based Recommendations")
        st.write("""
        Content-based recommendations suggest papers similar to ones you're interested in.
        You can search by title, authors, or categories.
        """)
        
        # Tabs for different search methods
        tab1, tab2, tab3 = st.tabs(["Search by Title", "Search by Authors", "Search by Categories"])
        
        with tab1:
            st.subheader("Find Papers Similar to a Title")
            title_query = st.text_input("Enter a paper title (or part of it):", key="title_input")
            method = st.radio("Similarity Method:", ["Semantic (Transformer)", "TF-IDF"], key="title_method")
            
            if st.button("Get Recommendations", key="title_button"):
                if title_query:
                    with st.spinner("Finding similar papers..."):
                        method_param = 'semantic' if method == "Semantic (Transformer)" else 'tfidf'
                        recommendations = st.session_state.content_recommender.get_recommendations_by_title(
                            title_query, method=method_param, top_n=10
                        )
                    
                    if not recommendations.empty:
                        st.success(f"Found {len(recommendations)} similar papers!")
                        for _, paper in recommendations.iterrows():
                            display_paper_card(paper)
                    else:
                        st.error("No similar papers found. Try a different title.")
                else:
                    st.warning("Please enter a paper title.")
        
        with tab2:
            st.subheader("Find Papers by Authors")
            authors_query = st.text_input("Enter author names (comma separated):", key="authors_input")
            
            if st.button("Get Recommendations", key="authors_button"):
                if authors_query:
                    authors_list = [author.strip() for author in authors_query.split(",")]
                    with st.spinner("Finding papers by authors..."):
                        recommendations = st.session_state.content_recommender.get_recommendations_by_authors(
                            authors_list, top_n=10
                        )
                    
                    if not recommendations.empty:
                        st.success(f"Found {len(recommendations)} papers by these authors!")
                        for _, paper in recommendations.iterrows():
                            display_paper_card(paper)
                    else:
                        st.error("No papers found by these authors. Try different names.")
                else:
                    st.warning("Please enter author names.")
        
        with tab3:
            st.subheader("Find Papers by Categories")
            categories_query = st.text_input("Enter categories (comma separated):", key="categories_input")
            
            if st.button("Get Recommendations", key="categories_button"):
                if categories_query:
                    categories_list = [category.strip() for category in categories_query.split(",")]
                    with st.spinner("Finding papers by categories..."):
                        recommendations = st.session_state.content_recommender.get_recommendations_by_categories(
                            categories_list, top_n=10
                        )
                    
                    if not recommendations.empty:
                        st.success(f"Found {len(recommendations)} papers in these categories!")
                        for _, paper in recommendations.iterrows():
                            display_paper_card(paper)
                    else:
                        st.error("No papers found in these categories. Try different categories.")
                else:
                    st.warning("Please enter categories.")
    
    # Collaborative Recommendations
    elif page == "Collaborative Recommendations":
        st.title("Collaborative Recommendations")
        st.write("""
        Collaborative recommendations suggest papers based on what similar users have liked.
        This system uses SVD and SVD++ algorithms to find patterns in user preferences.
        """)
        
        # Tabs for different collaborative methods
        tab1, tab2, tab3 = st.tabs(["Get Recommendations", "Rate Papers", "User ID Recommendations"])
        
        with tab1:
            st.subheader("Get Personalized Recommendations")
            model_type = st.radio("Model Type:", ["SVD", "SVD++"], key="collab_model")
            
            if st.button("Get Recommendations", key="collab_button"):
                with st.spinner("Finding personalized recommendations..."):
                    model = 'svd' if model_type == "SVD" else 'svdpp'
                    recommendations = st.session_state.collaborative_recommender.get_top_n_recommendations(
                        st.session_state.user_id, n=10, model=model
                    )
                
                if not recommendations.empty:
                    st.success(f"Found {len(recommendations)} personalized recommendations!")
                    for _, paper in recommendations.iterrows():
                        display_paper_card(paper)
                else:
                    st.error("No recommendations found. Try rating some papers first.")
        
        with tab2:
            st.subheader("Rate Papers to Improve Recommendations")
            
            # Show random papers to rate
            if st.button("Show Random Papers to Rate"):
                random_papers = st.session_state.papers_df.sample(5)
                for i, (_, paper) in enumerate(random_papers.iterrows()):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        display_paper_card(paper)
                    with col2:
                        rating = st.select_slider(
                            f"Rate Paper {i+1}",
                            options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                            value=3.0,
                            key=f"rating_{paper['id']}"
                        )
                        if st.button("Submit Rating", key=f"submit_{paper['id']}"):
                            # Add the rating to the user data
                            if st.session_state.collaborative_recommender.user_data is None:
                                st.session_state.collaborative_recommender.simulate_user_data()
                            
                            # Check if this paper is already rated by the user
                            user_ratings = st.session_state.collaborative_recommender.user_data
                            existing_rating = user_ratings[
                                (user_ratings['user_id'] == st.session_state.user_id) & 
                                (user_ratings['paper_id'] == paper['id'])
                            ]
                            
                            if existing_rating.empty:
                                # Add new rating
                                new_rating = pd.DataFrame({
                                    'user_id': [st.session_state.user_id],
                                    'paper_id': [paper['id']],
                                    'rating': [rating]
                                })
                                st.session_state.collaborative_recommender.user_data = pd.concat(
                                    [st.session_state.collaborative_recommender.user_data, new_rating],
                                    ignore_index=True
                                )
                            else:
                                # Update existing rating
                                st.session_state.collaborative_recommender.user_data.loc[
                                    (st.session_state.collaborative_recommender.user_data['user_id'] == st.session_state.user_id) & 
                                    (st.session_state.collaborative_recommender.user_data['paper_id'] == paper['id']),
                                    'rating'
                                ] = rating
                            
                            st.success(f"Rating submitted: {rating} stars")
                            
                            # Retrain the models with the new rating
                            with st.spinner("Updating recommendations..."):
                                st.session_state.collaborative_recommender.prepare_surprise_data()
                                st.session_state.collaborative_recommender.train_svd_model()
                                st.session_state.collaborative_recommender.train_svdpp_model()
                            
                            st.success("Recommendation models updated!")
        
        with tab3:
            st.subheader("Get Recommendations for Specific User ID")
            
            # Show available user IDs
            if st.session_state.collaborative_recommender.user_data is not None:
                available_users = st.session_state.collaborative_recommender.user_data['user_id'].unique().tolist()
                
                # Allow user to select a user ID
                selected_user_id = st.selectbox(
                    "Select a User ID:",
                    options=available_users,
                    index=0 if available_users else None,
                    key="selected_user_id"
                )
                
                # Allow custom user ID input
                custom_user_id = st.text_input("Or enter a custom User ID:", key="custom_user_id")
                
                # Use custom ID if provided, otherwise use selected ID
                user_id_to_use = custom_user_id if custom_user_id else selected_user_id
                
                # Model selection
                model_type = st.radio("Model Type:", ["SVD", "SVD++"], key="user_collab_model")
                
                if st.button("Get Recommendations", key="user_collab_button") and user_id_to_use:
                    with st.spinner(f"Finding recommendations for user {user_id_to_use}..."):
                        model = 'svd' if model_type == "SVD" else 'svdpp'
                        recommendations = st.session_state.collaborative_recommender.get_top_n_recommendations(
                            user_id_to_use, n=10, model=model
                        )
                    
                    if not recommendations.empty:
                        st.success(f"Found {len(recommendations)} personalized recommendations for user {user_id_to_use}!")
                        for _, paper in recommendations.iterrows():
                            display_paper_card(paper)
                    else:
                        st.error(f"No recommendations found for user {user_id_to_use}. Try a different user ID or rate some papers first.")
            else:
                st.warning("No user data available. Please generate user data first by rating some papers.")
    
    # Hybrid Recommendations
    elif page == "Hybrid Recommendations":
        st.title("Hybrid Recommendations")
        st.write("""
        Hybrid recommendations combine content-based and collaborative filtering approaches
        to provide more accurate and diverse paper suggestions.
        """)
        
        # Tabs for different hybrid methods
        tab1, tab2, tab3 = st.tabs([
            "Title + User Preferences", 
            "Categories + User Preferences",
            "Authors + User Preferences"
        ])
        
        with tab1:
            st.subheader("Find Papers Similar to a Title and Your Preferences")
            title_query = st.text_input("Enter a paper title (or part of it):", key="hybrid_title_input")
            content_weight = st.slider(
                "Content-Based Weight", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.1,
                key="title_weight"
            )
            
            if st.button("Get Recommendations", key="hybrid_title_button"):
                if title_query:
                    with st.spinner("Finding hybrid recommendations..."):
                        recommendations = st.session_state.hybrid_recommender.get_weighted_recommendations(
                            user_id=st.session_state.user_id,
                            paper_title=title_query,
                            content_weight=content_weight,
                            top_n=10
                        )
                    
                    if not recommendations.empty:
                        st.success(f"Found {len(recommendations)} hybrid recommendations!")
                        for _, paper in recommendations.iterrows():
                            display_paper_card(paper)
                    else:
                        st.error("No recommendations found. Try a different title or adjust the weights.")
                else:
                    st.warning("Please enter a paper title.")
        
        with tab2:
            st.subheader("Find Papers by Categories and Your Preferences")
            categories_query = st.text_input("Enter categories (comma separated):", key="hybrid_categories_input")
            content_weight = st.slider(
                "Content-Based Weight", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.7, 
                step=0.1,
                key="categories_weight"
            )
            
            if st.button("Get Recommendations", key="hybrid_categories_button"):
                if categories_query:
                    categories_list = [category.strip() for category in categories_query.split(",")]
                    with st.spinner("Finding hybrid recommendations..."):
                        recommendations = st.session_state.hybrid_recommender.get_recommendations_by_categories_and_user(
                            user_id=st.session_state.user_id,
                            categories=categories_list,
                            content_weight=content_weight,
                            top_n=10
                        )
                    
                    if not recommendations.empty:
                        st.success(f"Found {len(recommendations)} hybrid recommendations!")
                        for _, paper in recommendations.iterrows():
                            display_paper_card(paper)
                    else:
                        st.error("No recommendations found. Try different categories or adjust the weights.")
                else:
                    st.warning("Please enter categories.")
        
        with tab3:
            st.subheader("Find Papers by Authors and Your Preferences")
            authors_query = st.text_input("Enter author names (comma separated):", key="hybrid_authors_input")
            content_weight = st.slider(
                "Content-Based Weight", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.7, 
                step=0.1,
                key="authors_weight"
            )
            
            if st.button("Get Recommendations", key="hybrid_authors_button"):
                if authors_query:
                    authors_list = [author.strip() for author in authors_query.split(",")]
                    with st.spinner("Finding hybrid recommendations..."):
                        recommendations = st.session_state.hybrid_recommender.get_recommendations_by_authors_and_user(
                            user_id=st.session_state.user_id,
                            authors=authors_list,
                            content_weight=content_weight,
                            top_n=10
                        )
                    
                    if not recommendations.empty:
                        st.success(f"Found {len(recommendations)} hybrid recommendations!")
                        for _, paper in recommendations.iterrows():
                            display_paper_card(paper)
                    else:
                        st.error("No recommendations found. Try different authors or adjust the weights.")
                else:
                    st.warning("Please enter author names.")
    
    # Chat with RAG Assistant
    elif page == "Chat with RAG Assistant":
        st.title("Chat with Research Assistant")
        st.write("""
        Ask questions about research papers or topics, and get personalized recommendations.
        The assistant uses Retrieval-Augmented Generation (RAG) to provide accurate information.
        """)
        
        # Check if Gemini API key is set
        if not os.getenv("GEMINI_API_KEY"):
            st.warning("""
            Gemini API key not found. Please create a .env file with your GEMINI_API_KEY to enable the chat assistant.
            
            Example:
            ```
            GEMINI_API_KEY=your_api_key_here
            ```
            """)
            
            # Provide a text area for the API key
            api_key = st.text_input("Or enter your Gemini API key here:", type="password")
            if api_key and st.button("Set API Key"):
                os.environ["GEMINI_API_KEY"] = api_key
                st.session_state.rag_system.gemini_api_key = api_key
                st.session_state.rag_system.gemini_model = genai.GenerativeModel('gemini-pro')
                st.success("API key set successfully!")
                st.rerun()
        
        # Display chat history
        st.subheader("Conversation")
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-user">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-assistant">
                    <strong>Assistant:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        st.text_input(
            "Ask something about research papers:",
            key="chat_input",
            on_change=handle_chat_input
        )
        
        # Example questions
        st.subheader("Example Questions")
        example_questions = [
            "What are the latest papers on transformer models?",
            "Can you recommend papers about reinforcement learning?",
            "Who are the top authors in computer vision?",
            "Find papers similar to 'Attention is All You Need'",
            "What are the most popular categories in the dataset?"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"example_{hash(question)}"):
                st.session_state.chat_input = question
                handle_chat_input()
                st.rerun()

if __name__ == "__main__":
    main()
