# app1.py

# Import necessary libraries
import streamlit as st              # Web application framework
import pandas as pd                 # Data manipulation and analysis
import numpy as np                  # Numerical operations
import pickle                       # Loading/saving Python objects
import requests                     # Making HTTP requests
from PIL import Image              # Image processing
from io import BytesIO             # Binary I/O operations
import os                          # Operating system interface
from fuzzywuzzy import process     # String matching and comparison

# Suppress warning messages for cleaner output
import warnings
warnings.filterwarnings('ignore')

# TMDb API key for fetching movie information and posters
TMDB_API_KEY = '165b99f66e545cf009e58bac55519081'

# Cache the model loading to improve performance
@st.cache_resource
def load_models():
    """
    Load pre-computed similarity matrix and movie mappings
    
    Returns:
        tuple: Contains similarity matrix and movie mapping dictionaries
    """
    # Load the item similarity matrix from pickle file
    with open(r'C:\Users\anujp\OneDrive\Desktop\MovieRecommendations\models\collaborative_filtering\item_based\item_similarity.pkl', 'rb') as f:
        item_similarity_df = pickle.load(f)
    
    # Load movie ID to title mapping
    with open(r'C:\Users\anujp\OneDrive\Desktop\MovieRecommendations\models\collaborative_filtering\item_based\movie_id_to_title.pkl', 'rb') as f:
        movie_id_to_title = pickle.load(f)
    
    # Create reverse mapping (title to ID) for easier lookup
    title_to_movie_id = {title: movie_id for movie_id, title in movie_id_to_title.items()}
    
    return item_similarity_df, movie_id_to_title, title_to_movie_id

def fetch_poster_tmdb(movie_title, api_key):
    """
    Fetch movie poster and overview from TMDb API
    
    Args:
        movie_title (str): Name of the movie
        api_key (str): TMDb API key
    
    Returns:
        tuple: Contains poster URL and movie overview
    """
    try:
        # Remove year from movie title for better search results
        movie_title = movie_title.split('(')[0].strip()
        
        # Prepare API request parameters
        search_url = 'https://api.themoviedb.org/3/search/movie'
        params = {
            'api_key': api_key,
            'query': movie_title,
            'language': 'en-US',
            'include_adult': 'false'
        }
        
        # Make API request and parse response
        response = requests.get(search_url, params=params)
        data = response.json()
        
        # Extract poster and overview if available
        if data.get('results'):
            movie = data['results'][0]
            poster_path = movie.get('poster_path')
            if poster_path:
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                return poster_url, movie.get('overview', 'No overview available')
        return None, None
    except Exception as e:
        st.warning(f"Error fetching movie data: {str(e)}")
        return None, None

def get_similar_movies(movie_title, item_similarity_df, movie_id_to_title, title_to_movie_id, num_similar=5):
    """
    Find similar movies based on pre-computed similarity matrix
    
    Args:
        movie_title (str): Title of the reference movie
        item_similarity_df (DataFrame): Similarity matrix
        movie_id_to_title (dict): Mapping from movie ID to title
        title_to_movie_id (dict): Mapping from title to movie ID
        num_similar (int): Number of similar movies to return
    
    Returns:
        list: Similar movies with their similarity scores
    """
    # Check if movie exists in database
    if movie_title not in title_to_movie_id:
        return []
        
    # Get movie ID and check if it exists in similarity matrix
    movie_id = title_to_movie_id[movie_title]
    if movie_id not in item_similarity_df.index:
        return []
    
    # Get similarity scores and sort them
    sim_scores = item_similarity_df[movie_id]
    sim_scores = sim_scores.drop(movie_id, errors='ignore')  # Remove self-similarity
    sim_scores = sim_scores.sort_values(ascending=False)
    top_similar_ids = sim_scores.head(num_similar).index
    
    # Create list of similar movies with their similarity scores
    similar_movies = []
    for mid in top_similar_ids:
        title = movie_id_to_title[mid]
        similarity = sim_scores[mid]
        similar_movies.append((title, similarity))
    
    return similar_movies

def find_closest_title(input_title, title_list, min_score=60):
    """
    Find the closest matching movie title using fuzzy matching
    
    Args:
        input_title (str): User input title
        title_list (list): List of all movie titles
        min_score (int): Minimum similarity score threshold
    
    Returns:
        str: Closest matching title or None if no match found
    """
    closest_match = process.extractOne(input_title, title_list)
    if closest_match and closest_match[1] >= min_score:
        return closest_match[0]
    return None

def main():
    """
    Main function to run the Streamlit application
    """
    # Set up the webpage
    st.title("🎬 Movie Recommendation System")
    st.write("Enter a movie name to get similar movies!")
    
    try:
        # Load the required data and models
        with st.spinner("Loading movie database..."):
            item_similarity_df, movie_id_to_title, title_to_movie_id = load_models()
            movie_titles = list(title_to_movie_id.keys())
        
        # Create movie selection interface
        movie_title_input = st.selectbox(
            "Select or type a movie title:",
            options=movie_titles
        )
        
        # Allow user to choose number of recommendations
        num_recommendations = st.slider("Number of recommendations:", 1, 10, 5)
        
        # Generate recommendations when button is clicked
        if st.button("Get Recommendations"):
            if movie_title_input:
                closest_title = find_closest_title(movie_title_input, movie_titles)
                
                if closest_title:
                    # Inform user if using a different title than input
                    if closest_title != movie_title_input:
                        st.info(f"Using '{closest_title}' for recommendations")
                    
                    # Display selected movie information
                    st.subheader("Selected Movie:")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        poster_url, overview = fetch_poster_tmdb(closest_title, TMDB_API_KEY)
                        if poster_url:
                            st.image(poster_url, width=200)
                        else:
                            st.write("Poster not available")
                    with col2:
                        st.write(f"**Title:** {closest_title}")
                        if overview:
                            st.write("**Overview:**", overview)
                    
                    # Get and display similar movies
                    similar_movies = get_similar_movies(
                        closest_title, 
                        item_similarity_df, 
                        movie_id_to_title, 
                        title_to_movie_id, 
                        num_recommendations
                    )
                    
                    # Display recommendations if found
                    if similar_movies:
                        st.subheader("Similar Movies:")
                        cols = st.columns(min(3, len(similar_movies)))
                        col_idx = 0
                        
                        # Display each similar movie
                        for title, similarity in similar_movies:
                            with cols[col_idx]:
                                st.write(f"**{title}**")
                                st.write(f"Similarity: {similarity:.2f}")
                                
                                # Fetch and display movie poster
                                poster_url, overview = fetch_poster_tmdb(title, TMDB_API_KEY)
                                if poster_url:
                                    st.image(poster_url)
                                    with st.expander("Overview"):
                                        st.write(overview)
                                else:
                                    st.write("Poster not available")
                                
                                col_idx = (col_idx + 1) % len(cols)
                    else:
                        st.warning("No similar movies found.")
                else:
                    st.error("Movie not found in our database.")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please try again or contact support if the problem persists.")

# Entry point of the application
if __name__ == '__main__':
    main()