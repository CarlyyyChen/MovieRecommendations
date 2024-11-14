from surprise import Dataset, Reader, KNNWithMeans, accuracy
from surprise.model_selection import train_test_split

# TODO: import merged_data from processed data

reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(merged_data[['UserID','MovieID','Rating']], reader)

# Split into train and test sets
trainset, testset = train_test_split(data, test_size=0.25)

'''User based CF'''

sim_options_user_based = {
    "name": "cosine",
    "user_based": True,  # Compute similarities between users
}

# Initialize the KNNWithMeans algorithm with the similarity options
algo_user_based = KNNWithMeans(sim_options=sim_options_user_based)

# Train the algorithm on the trainset
algo_user_based.fit(trainset)

# Test the algorithm on the test set
predictions_user_based = algo_user_based.test(testset)

# Calculate accuracy (e.g., RMSE)
accuracy.rmse(predictions_user_based)
# RMSE: 0.9417

# Predict rating for a specific user and movie
user_id = 1
movie_id = 1193
prediction = algo_user_based.predict(user_id, movie_id)
print(prediction)

'''
Item based CF
We determine the similarities based on the movies
instead of looking at ratings from similar users
we look at ratings from similar movies 
'''

sim_options_movie_based = {
    "name": "cosine",
    "user_based": False,  # Compute similarities between movies
}

# Initialize the KNNWithMeans algorithm with the similarity options
algo_movie_based = KNNWithMeans(sim_options=sim_options_movie_based)

# Train the algorithm on the trainset
algo_movie_based.fit(trainset)

# Test the algorithm on the test set
predictions_movie_based = algo_movie_based.test(testset)

# Calculate accuracy (e.g., RMSE)
accuracy.rmse(predictions_movie_based)
# RMSE: 0.8963