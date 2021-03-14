import numpy as np
import random
import matplotlib.pyplot as plt

class KMeans():
    def __init__(self, n_clusters):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.n_clusters = n_clusters
        self.means = None
        self.assignments = None

    def update_assignments(self, features):
        #print("the means are", self.means)
        """
        for itc, sample in enumerate(features):
            optimal_distance = [999999, -1]
            for itc_mean, mean in enumerate(self.means):
                this_distance = np.linalg.norm(np.array(sample) - np.array(mean))
                if this_distance < optimal_distance[0]:
                    optimal_distance = [this_distance, itc_mean]
            self.assignments[itc] = optimal_distance[1] # is this what we really want to be assigning?
        """
        distances = np.array([np.linalg.norm(features-c, axis=1) for c in self.means])
        self.assignments = np.argmin(distances, axis=0)

    def update_means(self, features):
        print("the means are", self.means)
        """
        my_points = [None for _ in range(self.n_clusters)]
        for itc, mean in enumerate(self.means):
            my_points[itc] = [i for t, i in enumerate(features) if self.assignments[t] == itc]
            print(f"\npoints assigned to mean {itc} are\n", my_points)
        
        for itc, points in enumerate(my_points):
            self.means[itc] = (sum(points)/len(points)).tolist()"""
        for c in range(self.n_clusters):
            self.means[c] = np.mean(features[self.assignments == c], axis=0)

        print("the means now are", self.means)

    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """
        dimensions = features.shape
        
        def show():
            """
            if dimensions[1] == 2:
                fig = plt.figure()
                plt.title(f"Scatterplot after {i} iterations")
                plt.scatter(features[:, 0], features[:, 1], c=self.assignments, s=50, cmap='viridis')
                plt.scatter(self.means[0], self.means[1], c='black', s=200, alpha=0.5)
                plt.show()
            else:
                print("Unplottable, skipping")
            """
            pass
        
        self.assignments = [0 for _ in range(dimensions[0])]
        self.means = []
        for _ in range(self.n_clusters):
            self.means.append(random.choice(features))
        # means is now a list of the form [[x1,y1], [x2,y2]]

        for i in range(420):
            self.update_assignments(features)
            old_means = self.assignments
            self.update_means(features)
            if (old_means == self.assignments).all() and i > 0:
                print("Convergence, exiting after", i, "iterations")
                break

    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """
        return self.assignments