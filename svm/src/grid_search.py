'''
This class should implement Grid Search Cross Validation which is a hyperparameter
tuning approach. In fact, Grid Search CV will simply run many different classifiers
of the same type but with different hyperparameter settings.

Compared to Random Search CV the user can specify the parameters as well as the values
that should be used during the tuning process. The algorithm will then use ALL possible
combinations of parameters to run different models.

You need to use the parallelizer in the fit function of this class. The worker function
itself should run 20 fold cross validation. That means that you are running 20 trials.
'''
import itertools
from copy import deepcopy

from .parallelizer import Parallelizer
from .worker import run_experiment


class GridSearchCV:

    def __init__(self, estimator, tuned_parameters):
        '''
        This function should always get an estimator and a dictionary of parameters with possible
        values that the algorithm should use for hyperparameter tuning.

        :param estimator: The estimator that you want to use for this hyperparameter tuning session
        :param tuned_parameters: A dictionary that contains the parameter type as a key and a list
            of parameter values as value. To find an example of an example call you can take a look at
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        '''

        # This variable should contain the final results after fitting your dataset in an order manner
        self.cv_results = []
        self.estimator = estimator
        self.tuned_parameters = tuned_parameters

    def fit(self, inputs, targets):
        '''
        This function should create all possible mutations of the parameter specification and
        run all the different experiments in parallel. Finally, the variable self.cv_results
        should contain the final results for the hyperparameter tuning.

        First, implement the generate_all_permutations function to generate ALL permutations
        based on a dictionary that has parameter types as key and lists of possible parameter values
        for every single parameter type. After that you should start implementing this function.

        Simply loop over the list of ALL possible permutations that the generate_all_permutations
        function generated. For each iteration, generate a deepcopy of the self.estimator object
        and initialize the new estimator with the __init__() function. You can dynamically pass in
        a dictionary of arguments into a python function with the following piece of src:

        estimator.__init__(**param_config)

        For each iteration generate a new tuple consisting of your estimator, the param_config you
        used, the inputs and the targets. This should roughly look like this:

        tuple = (estimator, param_config, inputs, targets)

        Collect all the tuples and initialize a new Parallerlizer object. call the parallelize function
        and pass in the list of tuples.

        :param inputs: The input features/data that you want to use for classification
        :param targets: The targets/classes for the given input data
        '''

        perms = self.generate_all_permutations(self.tuned_parameters)

        tuplist = []

        for perm in perms:
            deep_est = deepcopy(self.estimator)
            deep_est.__init__(**perm)
            tup = (deep_est, perm, inputs, targets)
            tuplist.append(tup)
        
        self.cv_results = Parallelizer(run_experiment).parallelize(tuplist)

    def generate_all_permutations(self, tuned_parameters):
        '''
        This function will return all possible permutations for a given dictionary of
        parameter_type (keys) and parameter_values (values). The list should finally look
        like this:

        [
            {"kernel": "poly", "C": 10.0},
            {"kernel": "poly", "C": 1.0},
            {"kernel": "rbf", "C": 10.0},
            {"kernel": "rbf", "C": 1.0},
        ]

        :return: Returns all possible mutations as a list of dictionaries. Each dictionary should
            contain parameter_type and parameter_value pairs.
        '''

        return [dict(zip(tuned_parameters, y)) for y in itertools.product(*tuned_parameters.values())]