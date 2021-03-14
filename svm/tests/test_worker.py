import random

from sklearn import svm

from src.circle import load_circle
from src.worker import run_experiment


def test_worker():
    random.seed(1)
    inputs, targets = load_circle()
    params = {"kernel": "poly", "degree": 2}
    result = run_experiment(svm.SVC(kernel="poly", degree=2), params, inputs, targets)
    assert result[0] == params
    assert result[1] > 0.95
