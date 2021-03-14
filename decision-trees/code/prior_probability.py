import numpy as np

class PriorProbability():
    def __init__(self):
        """
        This is a simple classifier that only uses prior probability to classify 
        points. It just looks at the classes for each data point and always predicts
        the most common class.

        """
        self.most_common_class = None

    def fit(self, features, targets):
        """
        Implement a classifier that works by prior probability. Takes in features
        and targets and fits the features to the targets using prior probability.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N 
                examples.
        """

        counts = np.bincount(targets)
        self.most_common_class = np.argmax(counts)

    def predict(self, data):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        """

        n = data.shape[0]
        if self.most_common_class == 0:
            return np.zeros(n)
        elif self.most_common_class == 1:
            return np.ones(n)
        elif self.most_common_class == None:
            raise AssertionError("self.most_common_class is still None")
        else:
            raise AssertionError("Unknown self.most_common_class")


if __name__=="__main__":
    pp = PriorProbability()
    pp.fit(np.array([[1,2], [3,4]]), np.array([1, 0, 0, 0]))
    print(pp.most_common_class)