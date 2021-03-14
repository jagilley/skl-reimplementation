import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

def transform_data(features):
    """
    Data can be transformed before being put into a linear discriminator. If the data
    is not linearly separable, it can be transformed to a space where the data
    is linearly separable, allowing the perceptron algorithm to work on it. This
    function should implement such a transformation on a specific dataset (NOT 
    in general).

    Args:
        features (np.ndarray): input features
    Returns:
        transformed_features (np.ndarray): features after being transformed by the function
    """

    def cart2pol(my_row):
        x = my_row[0]
        y = my_row[1]
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return [rho, phi]

    #vfunc = np.vectorize(cart2pol)
    #transformed = vfunc(features)
    out = []
    #print(features, "\n\n")
    for row in features:
        out.append(cart2pol(row))
    out = np.array(out)
    #print("out is\n", out)
    """
    x = out[:,0]
    y = out[:,1]
    
    plt.scatter(x,y)
    plt.show()
    """
    return out

class Perceptron():
    def __init__(self, max_iterations=200):
        """
        This implements a linear perceptron for classification. A single
        layer perceptron is an algorithm for supervised learning of a binary
        classifier. The idea is to draw a linear line in the space that separates
        the points in the space into two partitions. Points on one side of the 
        line are one class and points on the other side are the other class.
        
        Note that label_for_example is either -1 or 1.

        Use only numpy to implement this algorithm. 

        Args:
            max_iterations (int): the perceptron learning algorithm stops after 
            this many iterations if it has not converged.

        """
        self.max_iterations = max_iterations
        self.w = None
        self.out = None

    def fit(self, features, targets):
        """
        Fit a single layer perceptron to features to classify the targets, which
        are classes (-1 or 1). This function should terminate either after
        convergence (dividing line does not change between interations) or after
        max_iterations (defaults to 200) iterations are done. Here is pseudocode for 
        the perceptron learning algorithm:
        
        begin initialize weights
            while not converged or not exceeded max_iterations
                for each example in features
                    if example is misclassified using weights
                    then weights = weights + example * label_for_example
            return weights
        end

        Args:
            features (np.ndarray): 2D array containing inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (saves model and training data internally)
        """
        w = np.array([1, 1, 1])
        old_w = np.array([42, 42, 42])
        itercount = 0
        while itercount < self.max_iterations:#not np.array_equal(w, old_w) or 
            for itc, example in enumerate(features):
                this_target = targets[itc]
                one_example = np.insert(example, 0, 1)
                prediction = 1 if w.dot(one_example)*this_target >= 0 else -1
                if prediction < 0:
                    old_w = w
                    w = w + one_example * this_target
            itercount += 1
        print("learned weights are", w)
        self.w = w

    def predict(self, features):
        """
        Given features, a 2D numpy array, use the trained model to predict target 
        classes. Call this after calling fit.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        """
        out_l = []
        one_features = np.concatenate((np.ones(features.shape[0])[:, np.newaxis], features), axis=1)
        for example in one_features:
            this_pred = 1 if self.w.dot(example) >= 0 else -1
            out_l.append(this_pred)
        self.out = out_l
        return np.array(out_l)

    def visualize(self, features, targets, my_title="untitled"):
        """
        This function should produce a single plot containing a scatter plot of the
        features and the targets, and the perceptron fit by the model should be
        graphed on top of the points.

        DO NOT USE plt.show() IN THIS FUNCTION.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (plots to the active figure)
        """
        try:
            import matplotlib.pyplot as plt
        except:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

        plt.figure(figsize=(6,4))
        #plt.contourf(self.out, cmap=plt.cm.Paired)
        plt.axis('off')
        plt.scatter(features[:, 0], features[:, 1], c=self.out)
        plt.title(my_title)
        plt.savefig(f'{my_title}.png')

if __name__=="__main__":
    from load_json_data import load_json_data
    for this_path in ["data/parallel_lines.json", "data/blobs.json", "data/circles.json", "data/crossing.json", "data/transform_me.json"]:
        features, targets = load_json_data(this_path)
        p = Perceptron(max_iterations=100)

        p.fit(features, targets)
        targets_hat = p.predict(features)
        p.visualize(features, targets, my_title=this_path.split("/")[-1])