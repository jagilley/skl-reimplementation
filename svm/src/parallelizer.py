'''
This helper class should make it easier for you to run a similar function call
in parallel. All you need to do is create one new object of this class and pass in
the function, which you want to run in parallel, to the constructor. After that you
simply need to call the parallelize function with a list of parameter tuples.
'''
from multiprocessing.dummy import Pool as ThreadPool

import warnings


class Parallelizer:

    def __init__(self, function):
        '''
        Initialize the Parallelizer with a function that you want to execute with different
        parameter settings in a new Thread. In our case, this will be the run_experiment function
        in the worker.py module.

        :param function: The function that you want to parallelize.
        '''
        self.function = function

    def parallelize(self, parameters):
        '''
        This function will run the function, that you specified in the constructor with the given
        list of specified parameter tuples. As a result you will get a list that contains multiple
        results. You need to keep in mind, that the result list might have another order that the
        parameters list.

        :param parameters: A list of tuples of parameters e.g. [(1_pa, 1_pb, 1_pc), (2_pa, 2_pb, 2_pc),]

        :return: A list that contains the return values for all the function calls
        '''

        warnings.simplefilter("ignore")

        # First we set the number of threads we want start
        # You can increase/decrease that number if you want
        n_threads = 16

        # Secondly, we need a list to store the results
        results = []

        # This creates a new pool for threads
        # Whenever there is space for new threads and there are new function calls in the queue
        # The Threadpool will call the function on a new thread
        pool = ThreadPool(n_threads)

        # Here is where the parallelization happens!
        # As already described, starmap will start a new Thread as soon as there is space in teh pool
        results += pool.starmap(self.function, parameters)

        return results
