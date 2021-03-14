import random

from sklearn import svm

from src.circle import load_circle
from src.experiment import run
from src.mnist import load_mnist


def test_experiment_run_mnist_grid():
    random.seed(1)

    NUMBER_OF_MNIST_SAMPLES = 500
    inputs, targets = load_mnist()
    inputs = inputs[:NUMBER_OF_MNIST_SAMPLES]
    targets = targets[:NUMBER_OF_MNIST_SAMPLES]

    results = run(svm.SVC(),
                 "grid_search",
                 {"kernel": ["linear", "poly", "rbf"], "C": [0.1, 1, 10]},
                 inputs,
                 targets)
    best_result = results[0][0]
    assert best_result["kernel"] == "rbf"
    assert best_result["C"] == 10.0

def jaspertest():
    NUMBER_OF_MNIST_SAMPLES = 500
    inputs, targets = load_mnist()
    inputs = inputs[:NUMBER_OF_MNIST_SAMPLES]
    targets = targets[:NUMBER_OF_MNIST_SAMPLES]

    results = run(svm.SVC(),
                 "grid_search",
                 {"kernel": ["linear"], "C": [1]},
                 inputs,
                 targets)
    best_result = results[0][0]
    print(results)

def test_experiment_run_polynomial_grid():
    inputs, targets = load_circle()
    random.seed(1)
    results = run(svm.SVC(),
                 "grid_search",
                 {"kernel": ["linear", "poly"], "degree": [1, 2, 3, 4, 5]},
                 inputs,
                 targets)
    best_result = results[0][0]
    assert best_result["kernel"] == "poly"
    assert best_result["degree"] == 2


def test_experiment_run_polynomial_random():
    inputs, targets = load_circle()
    random.seed(1)
    results = run(svm.SVC(),
                 "random_search",
                 {"kernel": ["linear", "poly"], "degree": [1, 2, 3, 4, 5]},
                 inputs,
                 targets,
                 n_iter=100)
    best_result = results[0][0]
    assert best_result["kernel"] == "poly"
    assert best_result["degree"] == 2