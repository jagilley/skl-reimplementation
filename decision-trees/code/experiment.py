from .decision_tree import DecisionTree
from .prior_probability import PriorProbability
from .metrics import precision_and_recall, confusion_matrix, f1_measure, accuracy
from .data import load_data, train_test_split

def run(data_path, learner_type, fraction):
    """
    This function walks through an entire machine learning workflow as follows:

        1. takes in a path to a dataset
        2. loads it into a numpy array with `load_data`
        3. instantiates the class used for learning from the data using learner_type (e.g
           learner_type is 'decision_tree', 'prior_probability')
        4. splits the data into training and testing with `train_test_split` and `fraction`.
        5. trains a learner using the training split with `fit`
        6. tests the trained learner using the testing split with `predict`
        7. evaluates the trained learner with precision_and_recall, confusion_matrix, and
           f1_measure

    Each run of this function constitutes a trial. Your learner should be pretty
    robust across multiple runs, as long as `fraction` is sufficiently high. See how
    unstable your learner gets when less and less data is used for training by
    playing around with `fraction`.

    IMPORTANT:
    If fraction == 1.0, then your training and testing sets should be exactly the
    same. This is so that the test cases are deterministic. The test case checks if you
    are fitting the training data correctly, rather than checking for generalization to
    a testing set.

    Args:
        data_path (str): path to csv file containing the data
        learner_type (str): either 'decision_tree' or 'prior_probability'. For each of these,
            the associated learner is instantiated and used for the experiment.
        fraction (float between 0.0 and 1.0): fraction of examples to be drawn for training

    Returns:
        confusion_matrix (np.array): Confusion matrix of learner on testing examples
        accuracy (np.float): Accuracy on testing examples using learner
        precision (np.float): Precision on testing examples using learner
        recall (np.float): Recall on testing examples using learner
        f1_measure (np.float): F1 Measure on testing examples using learner
    """

    # 2. loads dataset into a numpy array with `load_data`
    np_lines, np_targets, labels = load_data(data_path)

    # 3. instantiates the class used for learning from the data using learner_type (e.g
    #    learner_type is 'decision_tree', 'prior_probability')
    if learner_type == "prior_probability":
        learner = PriorProbability()
    elif learner_type == "decision_tree":
        learner = DecisionTree(labels)
    else:
        raise AssertionError(f"{learner_type} is not a valid learner_type")

    # 4. splits the data into training and testing with `train_test_split` and `fraction`
    if fraction != 1:
        train_features, train_targets, test_features, test_targets = train_test_split(np_lines, np_targets, fraction)
    else:
        train_features = np_lines
        test_features = np_lines
        train_targets = np_targets
        test_targets = np_targets

    # 5. trains a learner using the training split with `fit`
    learner.fit(train_features, train_targets)

    # 6. tests the trained learner using the testing split with `predict`
    predictions = learner.predict(train_features)

    # 7. evaluates the trained learner with precision_and_recall, confusion_matrix, and f1_measure
    c_matrix = confusion_matrix(test_targets, predictions)
    acc = accuracy(test_targets, predictions)
    precision, recall = precision_and_recall(test_targets, predictions)
    f1 = f1_measure(test_targets, predictions)

    # Order of these returns must be maintained
    return c_matrix, acc, precision, recall, f1