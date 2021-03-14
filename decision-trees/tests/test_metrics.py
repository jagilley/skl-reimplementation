import numpy as np
from .test_utils import make_fake_data

def test_accuracy():
    from sklearn.metrics import accuracy_score
    from code import accuracy

    y_true, y_pred = make_fake_data()
    _actual = accuracy_score(y_true, y_pred)
    _est = accuracy(y_true, y_pred)
    assert np.allclose(_actual, _est)

def test_f1_measure():
    from sklearn.metrics import f1_score
    from code import f1_measure

    y_true, y_pred = make_fake_data()
    _actual = f1_score(y_true, y_pred)
    _est = f1_measure(y_true, y_pred)
    assert np.allclose(_actual, _est)

def test_precision_and_recall():
    from sklearn.metrics import precision_score, recall_score
    from code import precision_and_recall

    y_true, y_pred = make_fake_data()
    _actual = [precision_score(y_true, y_pred), recall_score(y_true, y_pred)]
    _est = precision_and_recall(y_true, y_pred)
    assert np.allclose(_actual, _est)

def test_confusion_matrix():
    from sklearn.metrics import confusion_matrix as ref_confusion_matrix
    from code import confusion_matrix as est_confusion_matrix

    y_true, y_pred = make_fake_data()
    _actual = ref_confusion_matrix(y_true, y_pred)
    _est = est_confusion_matrix(y_true, y_pred)
    assert np.allclose(_actual, _est)
