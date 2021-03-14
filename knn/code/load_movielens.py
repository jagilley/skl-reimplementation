import csv
import numpy as np
import os
#import pandas as pd

def load_movielens_data(data_folder_path):
    """
    The MovieLens dataset is contained at data/ml-100k.zip. This function reads the
    unzipped content of the MovieLens dataset into a numpy array. The file to read in
    is called ```data/ml-100k/u.data``` The description of this dataset is:

    u.data -- The full u data set, 100000 ratings by 943 users on 1682 items.
              Each user has rated at least 20 movies.  Users and items are
              numbered consecutively from 1.  The data is randomly
              ordered. This is a tab separated list of 
	          user id | item id | rating | timestamp. 
              The time stamps are unix seconds since 1/1/1970 UTC

    Return a numpy array that has size 943x1682, with each item in the matrix (i, j)
    containing the rating user i had for item j. If a user i has no rating for item j,
    you should put 0 for that entry in the matrix.

    Args:
        data_folder_path {str}: Path to MovieLens dataset (given at data/ml-100).
    Returns:
        data {np.ndarray}: Numpy array of size 943x1682, with each item in the array
            containing the rating user i had for item j. If user i did not rate item j,
            the element (i, j) should be 0.
    """
    # This is the path to the file you need to load.
    data_file = os.path.join(data_folder_path, 'u.data')

    raw_datafile = np.loadtxt(data_file)

    #df = pd.DataFrame(raw_datafile, columns=["a", "b", "c", "d"])
    #print(df[df["b"] == 1682])
    #exit()

    #user_ids = raw_datafile[:,0]
    #uuids = np.unique(user_ids)[:, np.newaxis]
    uuids = np.zeros((943,))[:, np.newaxis]

    # get the outarr to have correct dimensions
    for _ in range(1681):
        zeroz = np.zeros((943,))[:, np.newaxis]
        uuids = np.hstack((uuids, zeroz))
    
    
    for row in raw_datafile:
        userid = int(row[0]) - 1
        itemid = int(row[1]) - 1
        rating = int(row[2])
        uuids[userid,itemid] = rating

    return uuids