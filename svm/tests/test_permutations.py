from src.grid_search import GridSearchCV


def test_grid_permutations():
    grid_search = GridSearchCV(None, None)

    permutations = grid_search.generate_all_permutations(
        {"kernel": ["poly", "linear", "rbf"], "C": [0.1, 1.0, 10.0], "gamma": [1.0, 10.0]})

    assert {"kernel": "poly", "C": 0.1, "gamma": 1.0} in permutations
    assert {"kernel": "linear", "C": 0.1, "gamma": 1.0} in permutations
    assert {"kernel": "rbf", "C": 0.1, "gamma": 1.0} in permutations
    assert {"kernel": "poly", "C": 1.0, "gamma": 1.0} in permutations
    assert {"kernel": "linear", "C": 1.0, "gamma": 1.0} in permutations
    assert {"kernel": "rbf", "C": 1.0, "gamma": 1.0} in permutations
    assert {"kernel": "poly", "C": 10.0, "gamma": 1.0} in permutations
    assert {"kernel": "linear", "C": 10.0, "gamma": 1.0} in permutations
    assert {"kernel": "rbf", "C": 1.0, "gamma": 1.0} in permutations
    assert {"kernel": "poly", "C": 0.1, "gamma": 10.0} in permutations
    assert {"kernel": "linear", "C": 0.1, "gamma": 10.0} in permutations
    assert {"kernel": "rbf", "C": 0.1, "gamma": 10.0} in permutations
    assert {"kernel": "poly", "C": 1.0, "gamma": 10.0} in permutations
    assert {"kernel": "linear", "C": 1.0, "gamma": 10.0} in permutations
    assert {"kernel": "rbf", "C": 1.0, "gamma": 10.0} in permutations
    assert {"kernel": "poly", "C": 10.0, "gamma": 10.0} in permutations
    assert {"kernel": "linear", "C": 10.0, "gamma": 10.0} in permutations
    assert {"kernel": "rbf", "C": 10.0, "gamma": 10.0} in permutations
