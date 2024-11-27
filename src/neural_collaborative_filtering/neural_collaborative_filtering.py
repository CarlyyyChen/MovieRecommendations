import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

TOP_N = 5
MODEL_PATH = "../../models/neural_collaborative_filtering/ncf_model.pkl"  # Path to save the model

# Function to save the model
def save_model(model, model_path):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

# Function to load the model
def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {model_path}")
    return model

# Function to load and merge datasets
def load_and_merge_data(movies_path, users_path, ratings_path):
    # Load the datasets
    movies = pd.read_csv(movies_path)
    users = pd.read_csv(users_path)
    ratings = pd.read_csv(ratings_path)

    # Merge the datasets
    merged_data = pd.merge(ratings, movies, on="MovieID")
    merged_data = pd.merge(merged_data, users, on="UserID")

    return merged_data

# Metrics for regression
def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Metrics for ranking
def precision_at_k(y_true, y_pred, k):
    """
    Precision@K: Fraction of relevant items in the top-K predictions.
    """
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    relevant_items = y_true[top_k_indices]
    return np.sum(relevant_items) / k

def recall_at_k(y_true, y_pred, k):
    """
    Recall@K: Fraction of relevant items among all relevant items.
    """
    if np.sum(y_true) == 0:  # No relevant items
        return 0.0
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    relevant_items = y_true[top_k_indices]
    return np.sum(relevant_items) / np.sum(y_true)

def ndcg_at_k(y_true, y_pred, k):
    """
    NDCG@K: Normalized Discounted Cumulative Gain.
    """
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    relevant_items = y_true[top_k_indices]

    # DCG: Discounted Cumulative Gain
    dcg = np.sum(relevant_items / np.log2(np.arange(2, k + 2)))

    # IDCG: Ideal DCG (sorted by relevance)
    ideal_relevance = np.sort(y_true)[::-1][:k]
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, k + 2)))

    return dcg / idcg if idcg > 0 else 0.0

# Custom Dataset for NCF
class NCFDataset(torch.utils.data.Dataset):
    def __init__(self, user_ids, movie_ids, ratings):
        self.user_ids = torch.tensor(user_ids.to_numpy(), dtype=torch.long)
        self.movie_ids = torch.tensor(movie_ids.to_numpy(), dtype=torch.long)
        self.ratings = torch.tensor(ratings.to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]

    # Define the NCF model
class NCFModel(torch.nn.Module):
    def __init__(self, num_users, num_movies, embed_dim):
        super(NCFModel, self).__init__()
        self.user_embedding = torch.nn.Embedding(num_users, embed_dim)
        self.movie_embedding = torch.nn.Embedding(num_movies, embed_dim)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, user_ids, movie_ids):
        user_embeds = self.user_embedding(user_ids)
        movie_embeds = self.movie_embedding(movie_ids)
        x = torch.cat([user_embeds, movie_embeds], dim=-1)
        return self.fc(x).squeeze() * 4 + 1 # scale to [1,5]

# new model
class NeuMFModel(nn.Module):
    def __init__(self, num_users, num_movies, embed_dim, mlp_layer_sizes=[16, 8, 4]):
        super(NeuMFModel, self).__init__()

        # GMF Embeddings
        self.gmf_user_embedding = nn.Embedding(num_users, embed_dim)
        self.gmf_movie_embedding = nn.Embedding(num_movies, embed_dim)

        # MLP Embeddings
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_layer_sizes[0] // 2)
        self.mlp_movie_embedding = nn.Embedding(num_movies, mlp_layer_sizes[0] // 2)

        # MLP Layers
        mlp_layers = []
        input_size = mlp_layer_sizes[0]
        for output_size in mlp_layer_sizes[1:]:
            mlp_layers.append(nn.Linear(input_size, output_size))
            mlp_layers.append(nn.ReLU())
            input_size = output_size
        self.mlp = nn.Sequential(*mlp_layers)

        # Final Layer
        self.final_layer = nn.Sequential(
            nn.Linear(embed_dim + mlp_layer_sizes[-1], 1),  # Combine GMF and MLP outputs
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, user_ids, movie_ids):
        # GMF Component
        gmf_user_embeds = self.gmf_user_embedding(user_ids)
        gmf_movie_embeds = self.gmf_movie_embedding(movie_ids)
        gmf_output = gmf_user_embeds * gmf_movie_embeds  # Element-wise product

        # MLP Component
        mlp_user_embeds = self.mlp_user_embedding(user_ids)
        mlp_movie_embeds = self.mlp_movie_embedding(movie_ids)
        mlp_input = torch.cat([mlp_user_embeds, mlp_movie_embeds], dim=-1)
        mlp_output = self.mlp(mlp_input)

        # Combine GMF and MLP
        combined = torch.cat([gmf_output, mlp_output], dim=-1)

        # Final Prediction
        output = self.final_layer(combined)

        # Scale output to range [1, 5]
        return output.squeeze() * 4 + 1

# Train the model
def train_model(model, data_loader, criterion, optimizer, epochs=5, k=5):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for user_ids, movie_ids, ratings in data_loader:
            optimizer.zero_grad()
            outputs = model(user_ids, movie_ids)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} | Loss: {total_loss / len(data_loader):.4f}")

# Validation with metrics
def evaluate_model(model, data_loader, k=TOP_N):
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for user_ids, movie_ids, ratings in data_loader:
            outputs = model(user_ids, movie_ids)
            all_targets.extend(ratings.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())

    # Calculate metrics
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    mae = calculate_mae(all_targets, all_predictions)
    rmse = calculate_rmse(all_targets, all_predictions)

    binary_targets = (all_targets > 4.5).astype(int)  # Example threshold: 4.5
    precision = precision_at_k(binary_targets, all_predictions, k)
    recall = recall_at_k(binary_targets, all_predictions, k)
    ndcg = ndcg_at_k(binary_targets, all_predictions, k)

    print(f"Evaluation | MAE: {mae:.4f} | RMSE: {rmse:.4f} | Precision@{k}: {precision:.4f} | "
          f"Recall@{k}: {recall:.4f} | NDCG@{k}: {ndcg:.4f}")

# Recommend movies for a user
def recommend_movies(model, user_id, all_movie_ids, k=TOP_N):
    model.eval()
    with torch.no_grad():
        user_tensor = torch.tensor([user_id] * len(all_movie_ids), dtype=torch.long)
        movie_tensor = torch.tensor(all_movie_ids, dtype=torch.long)
        predictions = model(user_tensor, movie_tensor)
    top_k_indices = predictions.argsort(descending=True)[:k]
    return top_k_indices.numpy(), predictions[top_k_indices].numpy()

def get_user_input():
    """Prompt the user to input a user ID and select a model."""
    print("\n--- Recommendation System ---")
    print("Enter 'q' at any time to quit.")

    user_id = input("Enter the User ID for recommendations (1-6040): ")
    if user_id.strip().lower() == 'q':
        return 'q'

    while not user_id.isdigit() or not (1 <= int(user_id) <= 6040):
        print("Invalid input. Please enter a numeric User ID between 1 and 6040 or 'q' to quit.")
        user_id = input("Enter the User ID for recommendations: ")
        if user_id.strip().lower() == 'q':
            return 'q'

    return user_id

# Main function
def main():
    # Paths to the datasets
    movies_path = "../../data/processed/movies.csv"
    users_path = "../../data/processed/users.csv"
    ratings_path = "../../data/processed/ratings.csv"

    # Load and merge data
    data = load_and_merge_data(movies_path, users_path, ratings_path)

    # Preprocessing: Convert UserID and MovieID to categorical codes
    data['UserID'] = data['UserID'].astype('category').cat.codes
    data['MovieID'] = data['MovieID'].astype('category').cat.codes

    # Extract features for NCF
    user_ids = data['UserID']
    movie_ids = data['MovieID']
    ratings = data['Rating']

    # Determine unique users and movies for embedding
    num_users = user_ids.nunique()
    num_movies = movie_ids.nunique()
    embed_dim = 50

    # Train-test split
    user_train, user_test, movie_train, movie_test, rating_train, rating_test = train_test_split(
        user_ids, movie_ids, ratings, test_size=0.2, random_state=42
    )

    # Create datasets and loaders
    train_dataset = NCFDataset(user_train, movie_train, rating_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_dataset = NCFDataset(user_test, movie_test, rating_test)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Check if the model already exists
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    else:
        # Initialize and train the model
        # model = NCFModel(num_users, num_movies, embed_dim)
        model = NeuMFModel(num_users, num_movies, embed_dim)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_model(model, train_loader, criterion, optimizer, epochs=100, k=5)
        save_model(model, MODEL_PATH)

    evaluate_model(model, test_loader, k=TOP_N)

    # Generate recommendations based on user input
    while True:
        user_id = get_user_input()
        if user_id == 'q':
            print("Exiting the Recommendation System!")
            break
        unique_movie_ids = movie_ids.unique()
        top_movies, predicted_ratings = recommend_movies(model, user_id, unique_movie_ids)

        print(f"\nTop {TOP_N} recommended movies for User {user_id}:")
        for i, (movie, rating) in enumerate(zip(top_movies, predicted_ratings)):
            print(f"{i + 1}: MovieID {movie}, Predicted Rating: {rating:.2f}")

if __name__ == "__main__":
    main()
