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

    return data


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


def main():
    """
    Main function to load data, train and evaluate user-based and item-based models.
    """
    # Load the dataset
    data = load_data()

    # Split the dataset into training and test sets
    trainset, testset = train_test_split(data, test_size=0.25)

    # User-Based Collaborative Filtering
    print("\n--- User-Based Collaborative Filtering ---")
    algo_user = KNNWithMeans(sim_options=SIM_OPTIONS_USER_BASED)
    user_rmse, user_predictions = train_and_evaluate(algo_user, trainset, testset)
    print(f"User-Based Model RMSE: {user_rmse:.4f}")

    # Predict a rating for a specific user and movie
    user_id = 1
    movie_id = 1193
    user_prediction = algo_user.predict(user_id, movie_id)
    print(f"Predicted rating for User {user_id} on Movie {movie_id}: {user_prediction.est:.2f}")

    # Item-Based Collaborative Filtering
    print("\n--- Item-Based Collaborative Filtering ---")
    algo_item = KNNWithMeans(sim_options=SIM_OPTIONS_ITEM_BASED)
    item_rmse, item_predictions = train_and_evaluate(algo_item, trainset, testset)
    print(f"Item-Based Model RMSE: {item_rmse:.4f}")


if __name__ == "__main__":
    main()
