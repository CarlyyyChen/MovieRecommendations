import pandas as pd
import os
from surprise import dump, Dataset, Reader, KNNWithMeans, accuracy
from surprise.model_selection import train_test_split

# Constants
SIM_OPTIONS = {
    "user": {
        "name": "cosine",
        "user_based": True,  # Compute similarities between users
    },
    "item": {
        "name": "cosine",
        "user_based": False,  # Compute similarities between items (movies)
    },
}
MODEL_PATHS = {
    "user": '../../models/collaborative_filtering/user_based_model.pkl',
    "item": '../../models/collaborative_filtering/item_based_model.pkl',
}
TOP_N = 5  # Number of recommendations to display


def save_model(algo, filename):
    """Save the trained model to a file."""
    dump.dump(filename, algo=algo)
    print(f"Model saved to {filename}")


def load_model(filename):
    """Load a trained model from a file."""
    if os.path.exists(filename):
        _, algo = dump.load(filename)
        print(f"Model loaded from {filename}")
        return algo
    print(f"No saved model found at {filename}.")
    return None


def load_data():
    """Load and preprocess the dataset from CSV files."""
    movies = pd.read_csv('../../data/processed/movies.csv')
    users = pd.read_csv('../../data/processed/users.csv')
    ratings = pd.read_csv('../../data/processed/ratings.csv')
    merged_data = pd.merge(ratings, movies, on='MovieID')
    merged_data = pd.merge(merged_data, users, on='UserID')

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(merged_data[['UserID', 'MovieID', 'Rating']], reader)

    return data, movies, ratings


def evaluate_model(algo, testset, model_name):
    """Evaluate a collaborative filtering model on a test set."""
    print(f"\n--- Evaluating {model_name} Collaborative Filtering Model ---")
    predictions = algo.test(testset)
    return accuracy.rmse(predictions, verbose=True)


def recommend_top_movies(algo, user_id, movies, ratings, top_n=TOP_N):
    """Recommend top N movies for a given user based on the predicted ratings."""
    all_movie_ids = set(movies['MovieID'])
    rated_movie_ids = set(ratings[ratings['UserID'] == user_id]['MovieID'])
    unrated_movie_ids = all_movie_ids - rated_movie_ids

    predictions = [
        (movie_id, algo.predict(user_id, movie_id).est) for movie_id in unrated_movie_ids
    ]
    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    top_movie_ids = [movie_id for movie_id, _ in sorted_predictions[:top_n]]

    return movies[movies['MovieID'].isin(top_movie_ids)]['Title'].tolist()


def get_user_input():
    """Prompt the user to input a user ID and select a model."""
    print("\n--- Recommendation System ---")
    print("Enter 'q' at any time to quit.")

    user_id = input("Enter the User ID for recommendations (1-6040): ")
    if user_id.strip().lower() == 'q':
        return None, 'q'

    while not user_id.isdigit() or not (1 <= int(user_id) <= 6040):
        print("Invalid input. Please enter a numeric User ID between 1 and 6040 or 'q' to quit.")
        user_id = input("Enter the User ID for recommendations: ")
        if user_id.strip().lower() == 'q':
            return None, 'q'

    model_choice = input("Select a model for recommendations (user/item): ").strip().lower()
    if model_choice == 'q':
        return None, 'q'

    while model_choice not in ['user', 'item']:
        print("Invalid choice. Please select either 'user' or 'item', or 'q' to quit.")
        model_choice = input("Select a model for recommendations (user/item): ").strip().lower()

    return int(user_id), model_choice


def train_or_load_model(trainset, model_type):
    """Load or train a collaborative filtering model."""
    filename = MODEL_PATHS[model_type]
    algo = load_model(filename)

    if algo is None:
        print(f"\n--- Training {model_type.capitalize()}-Based Collaborative Filtering Model ---")
        algo = KNNWithMeans(sim_options=SIM_OPTIONS[model_type])
        algo.fit(trainset)
        save_model(algo, filename)

    return algo


def main():
    """Main function to load data, train models, evaluate, and interact with the user."""
    data, movies, ratings = load_data()
    trainset, testset = train_test_split(data, test_size=0.25)

    algo_user = train_or_load_model(trainset, "user")
    algo_item = train_or_load_model(trainset, "item")

    evaluate_model(algo_user, testset, "User-Based")
    evaluate_model(algo_item, testset, "Item-Based")

    while True:
        user_id, model_choice = get_user_input()
        if model_choice == 'q':
            print("Exiting the Recommendation System!")
            break

        algo = algo_user if model_choice == 'user' else algo_item
        recommendations = recommend_top_movies(algo, user_id, movies, ratings)

        print(f"\nTop {TOP_N} recommended movies for User {user_id}:")
        for i, movie in enumerate(recommendations, start=1):
            print(f"{i}. {movie}")


if __name__ == "__main__":
    main()
