# Import required libraries
import streamlit as st              # For web application interface
import pandas as pd                 # For data manipulation and analysis
import numpy as np                  # For numerical operations
import pickle                       # For loading saved model files
import requests                     # For making HTTP requests to TMDb API
from PIL import Image              # For image processing
from io import BytesIO             # For handling binary data
import os                          # For file and directory operations
from sklearn.model_selection import train_test_split  # For splitting dataset

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# TMDb API Configuration
TMDB_API_KEY = '165b99f66e545cf009e58bac55519081'  # API key for The Movie Database

def verify_api_key(api_key):
    """
    Verify if the TMDb API key is valid by making a test request
    
    Args:
        api_key (str): TMDb API key to verify
    
    Returns:
        bool: True if API key is valid, False otherwise
    """
    test_url = "https://api.themoviedb.org/3/movie/550"
    params = {'api_key': api_key}
    try:
        response = requests.get(test_url, params=params)
        return response.status_code == 200
    except:
        return False

@st.cache_resource  # Cache the loaded model to avoid reloading
def load_model_and_mappings():
    """
    Load all necessary model files and mapping dictionaries
    
    Returns:
        tuple: Contains trained matrices (P, Q) and mapping dictionaries
    """
    try:
        # Load user latent factors matrix
        with open(r'C:\Users\anujp\OneDrive\Desktop\MovieRecommendations\src\data_preprocessing\P_trained.pkl', 'rb') as f:
            P_trained = pickle.load(f)
        
        # Load movie latent factors matrix
        with open(r'C:\Users\anujp\OneDrive\Desktop\MovieRecommendations\src\data_preprocessing\Q_trained.pkl', 'rb') as f:
            Q_trained = pickle.load(f)
        
        # Load user mapping dictionaries
        with open(r'C:\Users\anujp\OneDrive\Desktop\MovieRecommendations\src\data_preprocessing\user2idx.pkl', 'rb') as f:
            user2idx = pickle.load(f)
        
        with open(r'C:\Users\anujp\OneDrive\Desktop\MovieRecommendations\src\data_preprocessing\idx2user.pkl', 'rb') as f:
            idx2user = pickle.load(f)
        
        # Load movie mapping dictionaries
        with open(r'C:\Users\anujp\OneDrive\Desktop\MovieRecommendations\src\data_preprocessing\movie2idx.pkl', 'rb') as f:
            movie2idx = pickle.load(f)
        
        with open(r'C:\Users\anujp\OneDrive\Desktop\MovieRecommendations\src\data_preprocessing\idx2movie.pkl', 'rb') as f:
            idx2movie = pickle.load(f)
        
        return P_trained, Q_trained, user2idx, idx2user, movie2idx, idx2movie
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None

@st.cache_data  # Cache the data loading to avoid reloading
def load_and_prepare_data():
    """
    Load and prepare the dataset for recommendation
    
    Returns:
        tuple: Contains original dataframe and prepared training data
    """
    # Load the original dataset
    df = pd.read_csv(r"C:/Users/anujp/OneDrive/Desktop/MovieRecommendations/data/Final_data/Final_data.csv")
    
    # Split data into training and testing sets
    train_data, _ = train_test_split(df, test_size=0.2, random_state=42)
    train_data = train_data.reset_index(drop=True)
    
    # Create user and movie ID mappings
    user_ids = df['UserID'].unique()
    movie_ids = df['MovieID'].unique()
    
    user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie2idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    
    # Add mapped indices to training data
    train_data['user_idx'] = train_data['UserID'].map(user2idx)
    train_data['movie_idx'] = train_data['MovieID'].map(movie2idx)
    
    return df, train_data

def fetch_movie_info(movie_title, api_key):
    """
    Fetch movie information from TMDb API
    
    Args:
        movie_title (str): Movie title to search for
        api_key (str): TMDb API key
    
    Returns:
        dict: Movie information including poster URL and details
    """
    try:
        # Clean movie title by removing year and extra spaces
        clean_title = movie_title.split('(')[0].strip()
        
        # Prepare API request
        search_url = 'https://api.themoviedb.org/3/search/movie'
        params = {
            'api_key': api_key,
            'query': clean_title,
            'language': 'en-US',
            'include_adult': 'false'
        }
        
        # Make API request
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Process API response
        if data.get('results') and len(data['results']) > 0:
            movie = data['results'][0]
            poster_path = movie.get('poster_path')
            
            return {
                'poster_url': f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None,
                'overview': movie.get('overview', 'No overview available'),
                'release_date': movie.get('release_date', 'Release date unknown'),
                'vote_average': movie.get('vote_average', 'Rating not available'),
                'original_title': movie.get('original_title', clean_title)
            }
        
        return None
    
    except Exception as e:
        st.warning(f"Error fetching movie data: {str(e)}")
        return None

def recommend_movies(user_id, P, Q, user2idx, idx2movie, train_data, df, K=5):
    """
    Generate movie recommendations for a specific user
    
    Args:
        user_id (int): User ID to generate recommendations for
        P (numpy.array): User latent factors matrix
        Q (numpy.array): Movie latent factors matrix
        user2idx (dict): User ID to index mapping
        idx2movie (dict): Index to movie ID mapping
        train_data (pandas.DataFrame): Training dataset
        df (pandas.DataFrame): Original dataset
        K (int): Number of recommendations to generate
    
    Returns:
        list: List of recommended movies with predicted ratings
    """
    if user_id in user2idx:
        # Get user index and already rated movies
        user_idx = user2idx[user_id]
        user_rated_movies = train_data[train_data['user_idx'] == user_idx]['movie_idx'].tolist()
        
        # Calculate recommendation scores
        scores = np.dot(Q, P[user_idx, :])
        scores[user_rated_movies] = -np.inf
        top_k_movie_indices = np.argsort(-scores)[:K]
        
        # Generate recommendations
        recommendations = []
        for idx in top_k_movie_indices:
            movie_id = idx2movie.get(idx)
            if movie_id is not None:
                movie_row = df[df['MovieID'] == movie_id]
                if not movie_row.empty:
                    movie_title = movie_row['Title'].iloc[0]
                    predicted_rating = np.clip(np.dot(P[user_idx, :], Q[idx, :].T), 0.5, 5.0)
                    recommendations.append((movie_id, movie_title, predicted_rating))
        return recommendations
    return []

def display_user_history(df, user_id):
    """
    Display the rating history for a specific user
    
    Args:
        df (pandas.DataFrame): Dataset containing user ratings
        user_id (int): User ID to display history for
    """
    user_ratings = df[df['UserID'] == user_id].sort_values('Rating', ascending=False)
    if not user_ratings.empty:
        st.subheader("User's Rating History")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("Recent movies rated by user:")
            for _, row in user_ratings.head().iterrows():
                st.write(f"- {row['Title']}: {row['Rating']}⭐")
        with col2:
            avg_rating = user_ratings['Rating'].mean()
            num_ratings = len(user_ratings)
            st.write("Statistics:")
            st.write(f"Average rating: {avg_rating:.2f}⭐")
            st.write(f"Total ratings: {num_ratings}")

def main():
    """
    Main function to run the Streamlit application
    """
    # Set up the web page
    st.title("🎬 Movie Recommendation System using stochastic gradient descent ")
    st.write("Get personalized movie recommendations based on your user ID!")
    
    # Verify TMDb API key
    if not verify_api_key(TMDB_API_KEY):
        st.error("Invalid TMDb API key. Please check your API configuration.")
        return
    
    try:
        # Load model and data
        with st.spinner("Loading model and data..."):
            P_trained, Q_trained, user2idx, idx2user, movie2idx, idx2movie = load_model_and_mappings()
            df, train_data = load_and_prepare_data()
        
        # Get user input
        user_id_input = st.text_input("Enter UserID:", value="1")
        num_recommendations = st.slider("Number of recommendations:", 1, 10, 5)
        
        if st.button("Get Recommendations"):
            try:
                # Process user input
                user_id = int(user_id_input)
                if user_id not in user2idx:
                    st.error(f"User ID {user_id} not found in the dataset.")
                    return
                
                # Show user's rating history
                display_user_history(df, user_id)
                
                # Generate recommendations
                with st.spinner("Generating recommendations..."):
                    recommendations = recommend_movies(
                        user_id=user_id,
                        P=P_trained,
                        Q=Q_trained,
                        user2idx=user2idx,
                        idx2movie=idx2movie,
                        train_data=train_data,
                        df=df,
                        K=num_recommendations
                    )
                
                # Display recommendations
                if recommendations:
                    st.header("📽️ Recommended Movies:")
                    cols = st.columns(min(5, num_recommendations))
                    
                    for idx, (movie_id, title, pred_rating) in enumerate(recommendations):
                        with cols[idx % len(cols)]:
                            st.subheader(title)
                            st.write(f"Predicted: {pred_rating:.1f}⭐")
                            
                            # Fetch and display movie information
                            movie_info = fetch_movie_info(title, TMDB_API_KEY)
                            if movie_info and movie_info['poster_url']:
                                st.image(movie_info['poster_url'])
                                with st.expander("Movie Details"):
                                    st.write(f"**Release:** {movie_info['release_date']}")
                                    st.write(f"**TMDb Rating:** {movie_info['vote_average']}⭐")
                                    st.write("**Overview:**")
                                    st.write(movie_info['overview'])
                            else:
                                st.warning("Poster not available")
                else:
                    st.warning("No recommendations available for this user.")
                    
            except ValueError:
                st.error("Please enter a valid UserID (integer).")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please try again or contact support if the problem persists.")

# Entry point of the application
if __name__ == '__main__':
    main()