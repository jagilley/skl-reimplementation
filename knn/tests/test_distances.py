from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
import code
import numpy as np

def test_euclidean_distances():
    x = np.random.rand(100, 100)
    y = np.random.rand(100, 100)
    _true = euclidean_distances(x, y)
    _est = code.euclidean_distances(x, y)
    assert (np.allclose(_true, _est))


def test_manhattan_distances():
    x = np.random.rand(100, 100)
    y = np.random.rand(100, 100)
    _true = manhattan_distances(x, y)
    _est = code.manhattan_distances(x, y)
    assert (np.allclose(_true, _est))


def test_cosine_distances():
    x = np.random.rand(100, 100)
    y = np.random.rand(100, 100)
    _true = cosine_distances(x, y)
    _est = code.cosine_distances(x, y)
    assert (np.allclose(_true, _est))