import numpy as np

def confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the confusion matrix. The confusion 
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    YOU DO NOT NEED TO IMPLEMENT CONFUSION MATRICES THAT ARE FOR MORE THAN TWO 
    CLASSES (binary).
    
    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """

    if predictions.shape[0] != actual.shape[0]:
        pshape = predictions.shape[0]
        ashape = actual.shape[0]
        raise ValueError(f"predictions and actual must be the same length! predictions len is {pshape} and actual len is {ashape}")

    """
    True negatives is where both actual && predictions == 0
    False Positives is where actual == 0 && predictions == 1
    False Negatives is where actual == 1 && predictions == 0
    True Positives is where both actual && predictions == 1
    """
    
    actual_true_indices = np.where(actual==True)
    actual_false_indices = np.where(actual==False)
    
    actual_trues = predictions[actual_true_indices]
    actual_falses = predictions[actual_false_indices]

    true_positives = np.where(actual_trues==True)[0]
    true_negatives = np.where(actual_falses==False)[0]
    false_positives = np.where(actual_falses==True)[0]
    false_negatives = np.where(actual_trues==False)[0]

    true_positives = true_positives.shape[0]
    true_negatives = true_negatives.shape[0]
    false_positives = false_positives.shape[0]
    false_negatives = false_negatives.shape[0]

    c_matrix = np.array([
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ])

    return c_matrix

def accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the accuracy:

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    c_matrix = confusion_matrix(actual, predictions)
    true_positives = c_matrix[1,1]
    true_negatives = c_matrix[0,0]
    return (true_positives + true_negatives)/(actual.shape[0])

def precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        precision (float): precision
        recall (float): recall
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    c_matrix = confusion_matrix(actual, predictions)
    true_positives = c_matrix[1,1]
    true_negatives = c_matrix[0,0]
    false_positives = c_matrix[0,1]
    false_negatives = c_matrix[1,0]

    precision = true_positives/(true_positives + false_positives)
    recall = true_positives/(true_positives + false_negatives)

    return precision, recall

def f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the F1-measure:

   https://en.wikipedia.org/wiki/Precision_and_recall#F-measure

    Hint: implement and use the precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and 
        recall)
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    precision, recall = precision_and_recall(actual, predictions)
    return 2*((precision*recall)/(precision+recall))