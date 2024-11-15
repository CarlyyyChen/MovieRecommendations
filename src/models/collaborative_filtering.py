import pandas as pd
from surprise import Dataset, Reader, KNNWithMeans, accuracy
from surprise.model_selection import train_test_split

# Constants for similarity options
SIM_OPTIONS_USER_BASED = {
    "name": "cosine",
    "user_based": True  # Compute similarities between users
}

SIM_OPTIONS_ITEM_BASED = {
    "name": "cosine",
    "user_based": False  # Compute similarities between items (movies)
}


def load_data():
    """
    Load and preprocess the dataset from CSV files.

    Returns:
        data (Dataset): Surprise Dataset object for collaborative filtering.
    """
    # Load datasets
    movies = pd.read_csv('../../data/processed/movies.csv')
    users = pd.read_csv('../../data/processed/users.csv')
    ratings = pd.read_csv('../../data/processed/ratings.csv')

    # Merge dataframes to combine movie and user information with ratings
    merged_data = pd.merge(ratings, movies, on='MovieID')
    merged_data = pd.merge(merged_data, users, on='UserID')

    # Prepare Surprise Dataset
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(merged_data[['UserID', 'MovieID', 'Rating']], reader)

    return data, movies, ratings


def train_and_evaluate(algo, trainset, testset):
    """
    Train the collaborative filtering model and evaluate its performance.

    Args:
        algo (AlgoBase): A Surprise collaborative filtering algorithm.
        trainset (Trainset): The training data.
        testset (list): The test data.

    Returns:
        float: The RMSE of the predictions.
    """
    # Train the algorithm
    algo.fit(trainset)

    # Test the algorithm
    predictions = algo.test(testset)

    # Calculate RMSE
    rmse = accuracy.rmse(predictions, verbose=True)

    return rmse, predictions

def recommend_top_movies(algo, user_id, movies, ratings, top_n=5):
    """
    Recommend top N movies for a given user based on the predicted ratings.

    Args:
        algo (AlgoBase): The trained collaborative filtering model.
        user_id (int): The ID of the user for whom to recommend movies.
        movies (pd.DataFrame): DataFrame containing movie information (MovieID and Title).
        ratings (pd.DataFrame): DataFrame containing user ratings (UserID, MovieID, Rating).
        top_n (int): The number of top recommendations to return.

    Returns:
        list: A list of movie names recommended for the user.
    """
    # Get all movie IDs
    all_movie_ids = set(movies['MovieID'])

    # Get movies already rated by the user
    rated_movie_ids = set(ratings[ratings['UserID'] == user_id]['MovieID'])

    # Find movies the user has not rated
    unrated_movie_ids = all_movie_ids - rated_movie_ids

    # Predict ratings for unrated movies
    predictions = [
        (movie_id, algo.predict(user_id, movie_id).est) for movie_id in unrated_movie_ids
    ]

    # Sort predictions by rating in descending order
    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    # Get the top N movie IDs
    top_movie_ids = [movie_id for movie_id, _ in sorted_predictions[:top_n]]

    # Map movie IDs to names
    top_movie_names = movies[movies['MovieID'].isin(top_movie_ids)]['Title'].tolist()

    return top_movie_names


def get_user_input():
    """
    Prompt the user to input a user ID and select a model (user-based or item-based).

    Returns:
        user_id (int or None): The ID of the user for whom to generate recommendations, or None if quitting.
        model_choice (str): The selected model ('user' or 'item'), or 'q' to quit.
    """
    print("\n--- Recommendation System ---")
    print("Enter 'q' at any time to quit.")

    # Ask the user for a User ID
    user_id = input("Enter the User ID for recommendations (1-6040): ")
    if user_id.strip().lower() == 'q':
        return None, 'q'
    while not user_id.isdigit() or int(user_id) < 1 or int(user_id) > 6040:
        print("Invalid input. Please enter a numeric User ID between 1 and 6040 or 'q' to quit.")
        user_id = input("Enter the User ID for recommendations: ")
        if user_id.strip().lower() == 'q':
            return None, 'q'
    user_id = int(user_id)

    # Ask the user to select a model
    model_choice = input("Select a model for recommendations (user/item): ").strip().lower()
    if model_choice == 'q':
        return None, 'q'
    while model_choice not in ['user', 'item']:
        print("Invalid choice. Please select either 'user' or 'item', or 'q' to quit.")
        model_choice = input("Select a model for recommendations (user/item): ").strip().lower()
        if model_choice == 'q':
            return None, 'q'

    return user_id, model_choice


def main():
    """
    Main function to load data, train and evaluate user-based and item-based models.
    """
    # Load the dataset
    data, movies, ratings = load_data()

    # Split the dataset into training and test sets
    trainset, testset = train_test_split(data, test_size=0.25)

    # Train User-Based Collaborative Filtering Model
    print("\n--- Training User-Based Collaborative Filtering Model ---")
    algo_user = KNNWithMeans(sim_options=SIM_OPTIONS_USER_BASED)
    algo_user.fit(trainset)

    # Train Item-Based Collaborative Filtering Model
    print("\n--- Training Item-Based Collaborative Filtering Model ---")
    algo_item = KNNWithMeans(sim_options=SIM_OPTIONS_ITEM_BASED)
    algo_item.fit(trainset)

    # Start user interaction loop
    while True:
        # Get user input
        user_id, model_choice = get_user_input()

        # Quit if the user types 'q'
        if model_choice == 'q':
            print("Exiting the Recommendation System!")
            break

        # Generate recommendations based on the selected model
        top_n = 5  # Number of recommendations
        if model_choice == 'user':
            print("\n--- Recommendations from User-Based Model ---")
            recommendations = recommend_top_movies(
                algo_user, user_id, movies, ratings, top_n=top_n
            )
        else:
            print("\n--- Recommendations from Item-Based Model ---")
            recommendations = recommend_top_movies(
                algo_item, user_id, movies, ratings, top_n=top_n
            )

        # Print the recommendations
        print(f"\nTop {top_n} recommended movies for User {user_id}:")
        for i, movie in enumerate(recommendations, start=1):
            print(f"{i}. {movie}")


if __name__ == "__main__":
    main()
