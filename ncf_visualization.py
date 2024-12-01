import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
import requests
from sklearn.model_selection import train_test_split

# Constants
TOP_N = 5
TMDB_API_KEY = '165b99f66e545cf009e58bac55519081'

# NCF Dataset class - exact implementation
class NCFDataset(torch.utils.data.Dataset):
    def __init__(self, user_ids, movie_ids, ratings):
        self.user_ids = torch.tensor(user_ids.to_numpy(), dtype=torch.long)
        self.movie_ids = torch.tensor(movie_ids.to_numpy(), dtype=torch.long)
        self.ratings = torch.tensor(ratings.to_numpy(), dtype=torch.float32)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]

# NCF Model - exact implementation
class NCFModel(torch.nn.Module):
    def __init__(self, num_users, num_movies, embed_dim):
        super(NCFModel, self).__init__()
        self.user_embedding = torch.nn.Embedding(num_users, embed_dim)
        self.movie_embedding = torch.nn.Embedding(num_movies, embed_dim)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 2, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, user_ids, movie_ids):
        user_embeds = self.user_embedding(user_ids)
        movie_embeds = self.movie_embedding(movie_ids)
        x = torch.cat([user_embeds, movie_embeds], dim=-1)
        return self.fc(x).squeeze() * 4 + 1

# Evaluation metrics
def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def fetch_poster_tmdb(movie_title, api_key):
    try:
        movie_title = movie_title.split('(')[0].strip()
        search_url = 'https://api.themoviedb.org/3/search/movie'
        params = {
            'api_key': api_key,
            'query': movie_title,
            'language': 'en-US',
            'include_adult': 'false'
        }
        
        response = requests.get(search_url, params=params)
        data = response.json()
        
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

@st.cache_resource
def load_model_and_data():
    try:
        # Load data
        data = pd.read_csv('C:/Users/anujp/OneDrive/Desktop/MovieRecommendations/data/Final_data/Final_data.csv')
        
        # Load the saved model and mappings
        with open('C:/Users/anujp/OneDrive/Desktop/MovieRecommendations/models/neural_collaborative_filtering/ncf_model.pkl', 'rb') as f:
            saved_data = pickle.load(f)
        
        # Extract model configuration
        model_config = saved_data['model_config']
        
        # Initialize model with saved configuration
        model = NCFModel(
            num_users=model_config['num_users'],
            num_movies=model_config['num_movies'],
            embed_dim=model_config['embed_dim']
        )
        
        # Load the saved state
        model.load_state_dict(saved_data['model_state'])
        model.eval()
        
        # Apply the same mappings as during training
        user_mapping = saved_data['user_mapping']
        movie_mapping = saved_data['movie_mapping']
        
        # Create reverse mappings for convenience
        reverse_user_mapping = {v: k for k, v in user_mapping.items()}
        reverse_movie_mapping = {v: k for k, v in movie_mapping.items()}
        
        # Add mapping information to the data
        data['UserID_mapped'] = data['UserID'].map(reverse_user_mapping)
        data['MovieID_mapped'] = data['MovieID'].map(reverse_movie_mapping)
        
        return model, data, user_mapping, movie_mapping
        
    except FileNotFoundError:
        st.error("Model file not found. Please check the path.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

def recommend_movies(model, user_id, data, user_mapping, movie_mapping, k=TOP_N):
    """
    Generate movie recommendations using consistent mappings
    """
    model.eval()
    with torch.no_grad():
        # Map the user ID to the correct internal ID
        mapped_user_id = user_mapping.get(user_id)
        if mapped_user_id is None:
            st.error(f"User ID {user_id} not found in training data")
            return [], []
            
        # Get all unique movie IDs
        all_movie_ids = sorted(list(movie_mapping.values()))
        
        # Create tensors for prediction
        user_tensor = torch.tensor([mapped_user_id] * len(all_movie_ids), dtype=torch.long)
        movie_tensor = torch.tensor(all_movie_ids, dtype=torch.long)
        
        # Get predictions
        predictions = model(user_tensor, movie_tensor)
        
        # Get top k movies
        top_k_indices = predictions.argsort(descending=True)[:k]
        predicted_ratings = predictions[top_k_indices].numpy()
        
        # Map back to original movie IDs
        recommended_movies = []
        for idx in top_k_indices:
            movie_id = all_movie_ids[idx]
            original_movie_id = list(movie_mapping.keys())[list(movie_mapping.values()).index(movie_id)]
            recommended_movies.append(original_movie_id)
            
        return recommended_movies, predicted_ratings


def get_top_rated_movies(user_id, data):
    user_data = data[data['UserID'] == user_id]
    return user_data.sort_values(by='Rating', ascending=False).head(5)

def main():
    st.title("🎬 Neural Collaborative Filtering Movie Recommender")
    st.write("Enter a user ID to get personalized movie recommendations!")
    
    try:
        # Load model and data with mappings
        model, data, user_mapping, movie_mapping = load_model_and_data()
        
        if model is None:
            st.error("Failed to load model and data. Please check your files.")
            return
            
        # User input
        user_id = st.number_input(
            "Enter User ID (1-6040):",
            min_value=1,
            max_value=6040,
            value=6
        )
        
        if st.button("Get Recommendations"):
            # Show user's actual top rated movies
            st.subheader("User's Actual Top Rated Movies:")
            user_data = data[data['UserID'] == user_id]
            
            if user_data.empty:
                st.warning(f"No ratings found for User {user_id}")
            else:
                top_rated = user_data.sort_values('Rating', ascending=False).head(5)
                
                cols = st.columns(5)
                for idx, (_, row) in enumerate(top_rated.iterrows()):
                    with cols[idx]:
                        st.write(f"**{row['Title']}**")
                        st.write(f"Rating: {row['Rating']:.1f}")
                        poster_url, overview = fetch_poster_tmdb(row['Title'], TMDB_API_KEY)
                        if poster_url:
                            st.image(poster_url)
                        
                # Generate recommendations
                st.subheader("Model Recommendations:")
                recommended_movies, predicted_ratings = recommend_movies(
                    model, 
                    user_id, 
                    data, 
                    user_mapping, 
                    movie_mapping
                )
                
                if len(recommended_movies) > 0:
                    cols = st.columns(min(5, len(recommended_movies)))
                    for idx, (movie_id, rating) in enumerate(zip(recommended_movies, predicted_ratings)):
                        # Safely get movie data
                        movie_data = data[data['MovieID'] == movie_id]
                        if not movie_data.empty:
                            with cols[idx]:
                                title = movie_data.iloc[0]['Title']
                                st.write(f"**{title}**")
                                st.write(f"Predicted: {rating:.2f}")
                                
                                poster_url, overview = fetch_poster_tmdb(title, TMDB_API_KEY)
                                if poster_url:
                                    st.image(poster_url)
                                    if overview:
                                        with st.expander("Overview"):
                                            st.write(overview)
                else:
                    st.warning("No recommendations could be generated.")
                    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check if the user ID exists in the dataset.")
        st.write("Error details for debugging:", str(e))

if __name__ == '__main__':
    main()