import numpy as np 
from code import Perceptron, load_json_data, transform_data

def test_perceptron():
    features, targets = load_json_data('data/parallel_lines.json')
    
    p = Perceptron(max_iterations=100)

    p.fit(features, targets)
    targets_hat = p.predict(features)

    #your perceptron should fit this dataset perfectly

    assert np.allclose(targets, targets_hat)

def test_transform_data():
    features, targets = load_json_data('data/transform_me.json')
    features_transform = transform_data(features)

    p = Perceptron(max_iterations=100)

    p.fit(features_transform, targets)
    targets_hat = p.predict(features_transform)

    #your perceptron should fit this dataset perfectly after transforming the data
    assert np.allclose(targets, targets_hat)