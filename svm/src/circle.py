'''
This module helps to load a circle classification dataset.
The two classes are separated by a circle. Your
hyperparameter tuner should find an optimal classifier for this
task. The dataset looks like this where + and - denote samples
which belong to two different classes:

      +++++++
     +       +
    +   ---   +
    +   - -   +
    +   ---   +
     +       +
      +++++++
'''

import numpy as np


def load_circle():
    '''
    This function loads a circle classification dataset.

    :return: A polynomial circle classification dataset of degree 2.
        Your hyperparameter tuner should find a 2nd degree polynomial
        SVM for this dataset.
    '''
    x1 = np.random.uniform(low=-6.0, high=6.0, size=(2000,))
    x2 = np.random.uniform(low=-6.0, high=6.0, size=(2000,))

    y = random_circle(x1, x2)

    x1 = x1 / 6
    x2 = x2 / 6

    x = np.stack([x1, x2], axis=1)
    return x, y


def random_circle(x1, x2):
    '''
    Maps an x1, x2 value to a circular dataset

    :param x1: The x1 value that should be passed into the circle
    :param x2: Defines the height in the visualization

    :return: The classification value which is the result of the transformation
    '''

    function = np.sqrt(x1 ** 2 + x2 ** 2)

    return np.where(function > 3, 0, 1)
