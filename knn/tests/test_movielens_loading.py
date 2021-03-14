import numpy as np
from code import load_movielens_data
import os

def test_load_movielens_data():
    data = load_movielens_data(
        os.path.join('data',  'ml-100k')
    )
    assert (data.shape == (943, 1682) and np.count_nonzero(data) == 100000)
