import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

def adjusted_mutual_info(predicted_labels, predicted_targets):
    """
    Adjusted mutual info score is a metric used to judge the quality of a clustering
    output when ground truth labels are available. We're just wrapping it here so you
    can use it in your experiments. Here's the documentation:

    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html#sklearn.metrics.adjusted_mutual_info_score
    """
    return adjusted_mutual_info_score(predicted_labels, predicted_targets)