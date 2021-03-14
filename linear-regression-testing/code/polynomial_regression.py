import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

class PolynomialRegression():
    def __init__(self, degree):
        """
        Implement polynomial regression from scratch.
        
        This class takes as input "degree", which is the degree of the polynomial 
        used to fit the data. For example, degree = 2 would fit a polynomial of the 
        form:

            ax^2 + bx + c
        
        Your code will be tested by comparing it with implementations inside sklearn.
        DO NOT USE THESE IMPLEMENTATIONS DIRECTLY IN YOUR CODE. You may find the 
        following documentation useful:

        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

        Here are helpful slides:

        http://interactiveaudiolab.github.io/teaching/eecs349stuff/eecs349_linear_regression.pdf
    
        The internal representation of this class is up to you. Read each function
        documentation carefully to make sure the input and output matches so you can
        pass the test cases. However, do not use the functions numpy.polyfit or numpy.polval. 
        You should implement the closed form solution of least squares as detailed in slide 10
        of the lecture slides linked above.

        Usage:
            import numpy as np
            
            x = np.random.random(100)
            y = np.random.random(100)
            learner = PolynomialRegression(degree = 1)
            learner.fit(x, y) # this should be pretty much a flat line
            predicted = learner.predict(x)

            new_data = np.random.random(100) + 10
            predicted = learner.predict(new_data)

        Args:
            degree (int): Degree of polynomial used to fit the data.
        """
        self.degree = degree
        self.w = None
    
    def fit(self, features, targets):
        """
        Fit the given data using a polynomial. The degree is given by self.degree,
        which is set in the __init__ function of this class. The goal of this
        function is fit features, a 1D numpy array, to targets, another 1D
        numpy array.
        
        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing real-valued targets.
        Returns:
            None (saves model and training data internally)
        """
        """
        power_array = []
        this_deg = 0
        for _ in range(features.shape[1]):
            if this_deg <= self.degree:
                power_array.append(this_deg)
            else:
                power_array.append(self.degree)
            this_deg += 1
        print(power_array)
        features_raised = np.power(features, power_array)"""
        x_t = np.transpose(features)
        x_t_x = x_t*features # not sure if this is the correct matrix multiplication operator?
        x_t_x = x_t_x**(-1)
        x_t_y = x_t * targets
        w = x_t_x * x_t_y
        self.w = w
        print(w.shape)

    def predict(self, features):
        """
        Given features, a 1D numpy array, use the trained model to predict target 
        estimates. Call this after calling fit.

        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        """
        
        return features * self.w

    def visualize(self, features, targets):
        """
        This function should produce a single plot containing a scatter plot of the
        features and the targets, and the polynomial fit by the model should be
        graphed on top of the points.

        DO NOT USE plt.show() IN THIS FUNCTION. Instead, use plt.savefig().

        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing real-valued targets.
        Returns:
            None (plots to the active figure)
        """
        if self.w is None:
            raise AssertionError("Model has yet to be trained")

        preds = features * self.w
        plt.scatter(features, targets)
        plt.plot(features, preds, color='red')
        plt.savefig("output.png")

if __name__=="__main__":
    from generate_regression_data import generate_regression_data
    from metrics import mean_squared_error
    x, y = generate_regression_data(4, 100, amount_of_noise=0.1)
    random_ix = np.random.choice(x.shape[0], 50, replace=False)  
    x_training = x[random_ix]
    y_training = y[random_ix]
    mask = np.ones(len(x), np.bool)
    mask[random_ix] = 0
    x_testing = x[mask]
    y_testing = y[mask]
    print(len(x_testing))
    for degree_i in range(10):
        p = PolynomialRegression(degree_i)
        p.fit(x_training, y_training)
        y_hat = p.predict(x_testing[:50])
        print(y_hat)
        mse = mean_squared_error(y_training, y_hat)
        if degree_i == 1:
            print("vizzing")
            p.visualize(x[:50], y[:50])
        print(degree_i, mse)