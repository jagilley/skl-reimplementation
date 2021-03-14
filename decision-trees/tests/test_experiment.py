import numpy as np
from code import run
import os

datasets = [
    os.path.join('data', x)
    for x in os.listdir('data')
    if '.csv' in x
]

def xp_dataset_name(key):
    dataset = [d for d in datasets if key in d]
    if not dataset:
        raise ValueError('Dataset ' + key + ' cannot be found')
    return dataset[0]

def test_experiment_run_decision_tree():
    accuracies = {}
    for data_path in datasets:
        learner_type = 'decision_tree'
        confusion_matrix, accuracy, precision, recall, f1_measure = (
            run(data_path, learner_type, 1.0)
        )
        accuracies[data_path] = accuracy
    accuracy_goals = {
        xp_dataset_name('ivy-league.csv'): .95,
        xp_dataset_name('xor.csv'): 1.0,
        xp_dataset_name('candy-data.csv'): .75,
        xp_dataset_name('majority-rule.csv'): 1.0
    }
    for key in accuracy_goals:
        assert (accuracies[key] >= accuracy_goals[key])

def test_experiment_run_prior_probability():
    accuracies = {}
    for data_path in datasets:
        learner_type = 'prior_probability'
        confusion_matrix, accuracy, precision, recall, f1_measure = (
            run(data_path, learner_type, 1.0)
        )
        accuracies[data_path] = accuracy
    dataset = xp_dataset_name('ivy-league.csv')
    assert (accuracies[dataset] > .2)

def test_experiment_run_and_compare():
    for data_path in datasets:
        accuracies = {}
        learner_types = ['prior_probability', 'decision_tree']
        for learner_type in learner_types:
            accuracies[learner_type] = run(data_path, learner_type, 1.0)[1]
        if 'candy' in data_path or 'ivy' in data_path:
            assert (
                accuracies['decision_tree'] > accuracies['prior_probability']
            )
