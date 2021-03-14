import numpy as np 
from code import collaborative_filtering

def test_collaborative_filtering():
    user_ratings = np.array([
        [1, 0, 1, 2, 1],
        [2, 3, 0, 2, 1],
        [0, 1, 1, 4, 1],
        [2, 3, 3, 0, 3],
        [2, 3, 1, 2, 0],
        [4, 4, 0, 4, 4]
    ])
    real_ratings = np.array([
        [1, 3, 1, 2, 1],
        [2, 3, 1, 2, 1],
        [2, 1, 1, 4, 1],
        [2, 3, 3, 2, 3],
        [2, 3, 1, 2, 1],
        [4, 4, 1, 4, 4]
    ])
    predictions = collaborative_filtering(user_ratings, n_neighbors=5, aggregator='mode')
    print("preds are", predictions)
    assert (np.allclose(real_ratings, predictions))