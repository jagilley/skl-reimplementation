import numpy as np 

def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    xh, xv = X.shape
    yh, yv = Y.shape

    ret_arr = np.zeros((xh, yh))
    
    for itc1, row1 in enumerate(X):
        for itc2, row2 in enumerate(Y):
            ret_arr[itc1,itc2] = np.linalg.norm(row1-row2)

    return ret_arr


def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    xh, xv = X.shape
    yh, yv = Y.shape

    ret_arr = np.zeros((xh, yh))
    
    for itc1, row1 in enumerate(X):
        for itc2, row2 in enumerate(Y):
            ret_arr[itc1,itc2] = abs(row1-row2).sum()
            
    return ret_arr


def cosine_distances(X, Y):
    """Compute Cosine distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """
    xh, xv = X.shape
    yh, yv = Y.shape

    ret_arr = np.zeros((xh, yh))
    
    for itc1, row1 in enumerate(X):
        for itc2, row2 in enumerate(Y):
            ret_arr[itc1,itc2] = 1.0 - np.dot(row1, row2) / (np.linalg.norm(row1) * np.linalg.norm(row2))
            
    return ret_arr