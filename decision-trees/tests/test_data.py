import numpy as np
import random, string
import csv
from .test_utils import write_random_csv_file

def test_load_data():
    from code import load_data

    n_features = np.random.randint(5, 20)
    n_samples = np.random.randint(50, 150)
    features, targets, attribute_names = write_random_csv_file(n_features, n_samples)

    _features, _targets, _attribute_names = load_data('tests/test.csv')
    assert attribute_names == _attribute_names
    print(features, "\n\n", _features)
    assert np.allclose(features, _features) and np.allclose(targets, _targets)

def test_train_test_split():
    from code import train_test_split

    n_features = np.random.randint(5, 20)
    n_samples = np.random.randint(50, 150)
    features, targets, attribute_names = write_random_csv_file(n_features, n_samples)
    fraction = np.random.rand()

    output = train_test_split(features, targets, fraction)
    expected_train_size = int(n_samples * fraction)
    expected_test_size = n_samples - expected_train_size

    for o in output:
        assert o.shape[0] == expected_train_size or o.shape[0] == expected_test_size
