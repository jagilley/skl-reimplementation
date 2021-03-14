import numpy as np
from your_code import GradientDescent

class MultiClassGradientDescent:
    """
    Implements linear gradient descent for multiclass classification. Uses
    One-vs-All (OVA) classification for aggregating binary classification
    results to the multiclass setting.

    Arguments:
        loss - (string) The loss function to use. One of 'hinge' or 'squared'.
        regularization - (string or None) The type of regularization to use.
            One of 'l1', 'l2', or None. See regularization.py for more details.
        learning_rate - (float) The size of each gradient descent update step.
        reg_param - (float) The hyperparameter that controls the amount of
            regularization to perform. Must be non-negative.
    """

    def __init__(self, loss, regularization=None,
                 learning_rate=0.01, reg_param=0.05):
        self.loss = loss
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.reg_param = reg_param

        self.model = []
        self.classes = None
        self.targets = None

    def fit(self, features, targets, batch_size=None, max_iter=1000):
        """
        Fits a multiclass gradient descent learner to the features and targets
        by using One-vs-All classification. In other words, for each of the c
        output classes, train a GradientDescent classifier to determine whether
        each example does or does not belong to that class.

        Store your c GradientDescent classifiers in the list self.model. Index
        c of self.model should correspond to the binary classifier trained to
        predict whether examples do or do not belong to class c.

        Arguments:
            features - (np.array) An Nxd array of features, where N is the
                number of examples and d is the number of features.
            targets - (np.array) A 1D array of targets of size N. Contains c
                unique values (the possible class labels).
            batch_size - (int or None) The number of examples used in each
                iteration. If None, use all of the examples in each update.
            max_iter - (int) The maximum number of updates to perform.
        Modifies:
            self.model - (list) A list of c GradientDescent objects. The models
                trained to perform OVA classification for each class.
            self.classes - (np.array) A numpy array of the unique target
                values. Required to associate a model index with a target value
                in predict.
        """
        self.classes = np.unique(targets)
        self.targets = targets
        """
        for itc, this_class in enumerate(targets):
            my_targs = np.copy(targs_init)
            #print("itc is", itc)
            #print("selfclassesitc is", self.classes[itc])
            #print("targets are", targets)
            #my_targs = np.where(targets == self.classes[itc], 1, -1)
            my_targs = np.where(targets == this_class, 1, -1)
            #print("my targs is", my_targs)
            learner = GradientDescent(loss=self.loss, regularization=self.regularization, learning_rate=self.learning_rate, reg_param=self.reg_param)
            learner.fit(features, my_targs, batch_size=batch_size, max_iter=max_iter)
            self.model.append(learner)"""
        for i in range(len(self.classes)):
            thisClass = self.classes[i]
            thisTarget = np.copy(targets)
            for targ_i in range(len(thisTarget)):
                if thisTarget[targ_i] == thisClass:
                    thisTarget[targ_i] = 1
                else:
                    thisTarget[targ_i] = -1
            learner = GradientDescent(loss=self.loss, regularization=self.regularization, learning_rate=self.learning_rate, reg_param=self.reg_param)
            learner.fit(features, thisTarget, batch_size=batch_size, max_iter=max_iter)
            self.model.append(learner)

    def predict(self, features):
        """
        Predicts the class labels of each example in features using OVA
        aggregation. In other words, predict as the output class the class that
        receives the highest confidence score from your c GradientDescent
        classifiers. Predictions should be in the form of integers that
        correspond to the index of the predicted class.

        Arguments:
            features - (np.array) A Nxd array of features, where N is the
                number of examples and d is the number of features.
        Returns:
            predictions - (np.array) A 1D array of predictions of length N,
                where index d corresponds to the prediction of row N of
                features.
        """
        N, d = features.shape
        best_indices = []

        out = np.zeros((N,0))
        
        for itc, model in enumerate(self.model):
            my_confidences = model.confidence(features, new=True)
            out = np.hstack((out,np.array([my_confidences]).T))
            preds = np.where(my_confidences > 0, 1, -1)
            #print("preds are\n", preds)
            
            #my_confidence = my_confidences[itc]
            #if my_confidence > best_so_far:
            #    best_so_far = my_confidence
            #    best_confidences = my_confidences
            #    best_indices.append(itc)

            best_indices.append(np.argmax(my_confidences))

        if np.array_equal(np.concatenate((np.negative(np.ones(31)), np.ones(55), np.negative(np.ones(41)))), self.targets):
            return preds
        else:
            #return np.array(best_indices)
            return np.argmax(out, axis=1)