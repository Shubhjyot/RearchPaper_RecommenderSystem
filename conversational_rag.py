import os
import pandas as pd
import numpy as np
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import time
import joblib

# Load environment variables
load_dotenv()

class ConversationalRAG:
    def __init__(self, processed_data=None, data_path='processed_data.pkl'):
        """Initialize the conversational RAG system with processed data."""
        if processed_data is not None:
            self.data = processed_data
        elif os.path.exists(data_path):
            self.data = joblib.load(data_path)
        else:
            raise FileNotFoundError(f"Processed data file {data_path} not found.")
        
        # Initialize the sentence transformer model
        self.transformer_model = None
        self.embeddings = None
        self.faiss_index = None
        
        # Initialize Gemini
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-pro')
        else:
            print("Warning: GEMINI_API_KEY not found in environment variables.")
            self.gemini_model = None
        
        # Conversation history
        self.conversation_history = []
    
    def build_vector_db(self, model_name='all-MiniLM-L6-v2'):
        """Build the vector database for RAG."""
        print(f"Building vector database with {model_name}...")
        start_time = time.time()
        
        # Initialize the transformer model
        self.transformer_model = SentenceTransformer(model_name)
        
        # Prepare text for embedding
        # Combine title, summary, authors, and categories for comprehensive context
        text_for_embedding = (
            self.data['title'] + ". " + 
            self.data['summary'] + ". Authors: " + 
            self.data['authors'] + ". Categories: " + 
            self.data['categories']
        )
        
        # Generate embeddings
        self.embeddings = self.transformer_model.encode(
            text_for_embedding.tolist(), 
            show_progress_bar=True,
            batch_size=32
        )
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(self.embeddings.astype(np.float32))
        
        print(f"Vector database built in {time.time() - start_time:.2f} seconds.")
        return self.faiss_index
    
    def retrieve_relevant_papers(self, query, top_k=5):
        """Retrieve relevant papers for a query."""
        if self.transformer_model is None or self.faiss_index is None:
            raise ValueError("Vector database not built. Run build_vector_db first.")
        
        # Encode the query
        query_embedding = self.transformer_model.encode([query]).astype(np.float32)
        
        # Search for similar papers
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Get the relevant papers
        relevant_papers = self.data.iloc[indices[0]]
        
        return relevant_papers
    
    def format_papers_for_context(self, papers):
        """Format papers for context in the RAG system."""
        context = ""
        for i, (_, paper) in enumerate(papers.iterrows()):
            context += f"Paper {i+1}:\n"
            context += f"Title: {paper['title']}\n"
            context += f"Authors: {paper['authors']}\n"
            context += f"Categories: {paper['categories']}\n"
            context += f"Summary: {paper['summary'][:300]}...\n"
            context += f"URL: {paper['pdf_url']}\n\n"
        
        return context
    
    def generate_response(self, user_query):
        """Generate a response using RAG."""
        if self.gemini_model is None:
            return "Gemini API key not configured. Please set the GEMINI_API_KEY environment variable."
        
        # Retrieve relevant papers
        relevant_papers = self.retrieve_relevant_papers(user_query)
        
        # Format papers for context
        context = self.format_papers_for_context(relevant_papers)
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_query})
        
        # Prepare the prompt
        system_prompt = """You are a helpful research assistant that helps users find academic papers.
        You have access to a database of arXiv papers. When responding to the user:
        1. Be concise and informative
        2. If you're recommending papers, include their titles and URLs
        3. If the user asks about a specific topic, focus on papers in that area
        4. Always provide clickable URLs when mentioning papers
        5. Don't make up information - only use the provided context
        """
        
        prompt = f"""
        Context information from paper database:
        {context}
        
        Based on the above context, please respond to the user's query: {user_query}
        """
        
        # Generate response
        response = self.gemini_model.generate_content([
            {"role": "system", "parts": [system_prompt]},
            {"role": "user", "parts": [prompt]}
        ])
        
        # Update conversation history
        self.conversation_history.append({"role": "assistant", "content": response.text})
        
        return response.text
    
    def save_models(self, output_dir='.'):
        """Save the trained models to files."""
        if self.embeddings is not None:
            np.save(f"{output_dir}/rag_embeddings.npy", self.embeddings)
            print(f"RAG embeddings saved to {output_dir}")
        
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, f"{output_dir}/rag_faiss_index.bin")
            print(f"RAG FAISS index saved to {output_dir}")
    
    def load_models(self, input_dir='.'):
        """Load the trained models from files."""
        # Load embeddings
        if os.path.exists(f"{input_dir}/rag_embeddings.npy"):
            self.embeddings = np.load(f"{input_dir}/rag_embeddings.npy")
            print(f"RAG embeddings loaded from {input_dir}")
        
        # Load FAISS index
        if os.path.exists(f"{input_dir}/rag_faiss_index.bin"):
            self.faiss_index = faiss.read_index(f"{input_dir}/rag_faiss_index.bin")
            print(f"RAG FAISS index loaded from {input_dir}")

if __name__ == "__main__":
    # Example usage
    rag = ConversationalRAG()
    rag.build_vector_db()
    rag.save_models()
    print("Conversational RAG system built and saved.")
