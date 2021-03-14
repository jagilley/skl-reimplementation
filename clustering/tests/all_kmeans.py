from code import KMeans, GMM
import numpy as np
from code import adjusted_mutual_info, generate_cluster_data
import os
import random
from scipy.stats import multivariate_normal
from itertools import permutations

np.random.seed(0)
random.seed(0)

def test_kmeans_spec():
    features, targets = generate_cluster_data(
                        n_samples=100,
                        n_features=2,
                        n_centers=2,
                        cluster_stds=.1
                    )
    model = KMeans(2)
    model.fit(features)
    assert (hasattr(model, 'means'))

def test_kmeans_on_generated():
    n_samples = [1000, 10000]
    n_centers = [2]
    stds = [.1]
    n_features = [1, 2, 4]

    for n in n_samples:
        for f in n_features:
            for c in n_centers:
                for s in stds:
                    features, targets = generate_cluster_data(
                        n_samples=n,
                        n_features=f,
                        n_centers=c,
                        cluster_stds=s
                    )
                    # make model and fit
                    model = KMeans(c)
                    model.fit(features)

                    means = model.means
                    orderings = permutations(means)
                    distance_to_true_means = []

                    actual_means = np.array([
                        features[targets == i, :].mean(axis=0) for i in range(targets.max() + 1)
                    ])
                    
                    for ordering in orderings:
                        _means = np.array(list(ordering))
                        
                        distance_to_true_means.append(
                            np.abs(_means - actual_means).sum()
                        )
                    
                    assert (min(distance_to_true_means) < 1e-1)

                    # predict and calculate adjusted mutual info
                    labels = model.predict(features)
                    acc = adjusted_mutual_info(targets, labels)
                    assert (acc >= .9)