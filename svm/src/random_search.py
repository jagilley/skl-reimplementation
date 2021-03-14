'''
This class should implement Random Search Cross Validation which is a hyperparameter
tuning approach. In fact, Random Search CV will simply run many different classifiers
of the same type but with different hyperparameter settings.

The user can only specify the parameters and the distributions that it should sample
from. Furthermore, the user needs to specify how many random experiments the algorithm
should run. For example, if you have parameter A with values [0, 1, 2] and parameter
B with values [4, 5, 6] and you want to explore 4 experiments, Random Search CV
might run the experiments with following parameter combinations: [0, 4], [0,5], [1, 5], [1, 6]

You need to use the parallelizer in the fit function of this class. The worker function
itself should run 20 fold cross validation. That means that you are running 20 trials.
'''

import random
from copy import deepcopy

from .parallelizer import Parallelizer
from .worker import run_experiment


class RandomSearchCV:

    def __init__(self, estimator, param_distributions, n_iter=5):
        '''
        This function should always get an estimator and a dictionary of parameters with possible
        values that the algorithm should use for hyperparameter tuning.

        :param estimator: The estimator that you want to use for this hyperparameter tuning session
        :param tuned_parameters: A dictionary that contains the parameter type as a key and a distribution of
            parameter values that the algorithm should sample from. To find an example of an example call
            you can take a look at:
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV
        :param n_iter: The number of random experiments the algorithm should run
        '''

        # This variable should contain the final results after fitting your dataset
        self.cv_results = []
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter

    def fit(self, inputs, targets):
        '''
        This function should create n_iter random mutations of the parameter specification and
        run all the different experiments in parallel. Finally, the variable self.cv_results
        should contain the final results for the hyperparameter tuning.

        Create a loop that iterates self.n_iter times. Create an inner loop that creates random
        configurations for your estimator based on the param_distributions dictionary that you got.

        After creating a new random param configuration, generate a deepcopy of the self.estimator object
        and initialize the new estimator with the __init__() function. You can dynamically pass in
        a dictionary of arguments into a python function with the following piece of src:

        estimator.__init__(**param_config)

        The variable param_config should look something like this {"kernel": "poly", "degree" : 2, "C": 10.0}.
        For each iteration over n_iter you want to generate a new random configuration based on the param_distribution.
        For each iteration generate a new tuple consisting of your estimator, the param_config you
        used, the inputs and the targets. This should roughly look like this:

        tuple = (estimator, param_config, inputs, targets)

        Collect all the tuples and initialize a new Parallerlizer object. call the parallelize function
        and pass in the list of tuples. You can create a new parallelizer with the worker function defined in
        worker.py like this:

        parallelizer = Parallelizer(run_experiment)

        :param inputs: The input features/data that you want to use for classification
        :param targets: The targets/classes for the given input data
        '''

        outlist = []

        for _ in range(self.n_iter):
            this_config = {}
            for key, val in self.param_distributions.items():
                this_config[key] = random.choice(val)
            deep_est = deepcopy(self.estimator)
            deep_est.__init__(**this_config)
            tup = (deep_est, this_config, inputs, targets)
            outlist.append(tup)
        
        self.cv_results = Parallelizer(run_experiment).parallelize(outlist)