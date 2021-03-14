import numpy as np
import pandas as pd
from matplotlib.pyplot import imread
import os

class DogsDataset:
    def __init__(self, path_to_dogsset, num_classes=10):
        """
        This is a class that loads DogSet into memory. Give it the path
        to the DogSet folder on your machine and it will load it when you
        initialize this object.

        Training examples are stored in `self.trainX` and `self.trainY`
        for the images and labels, respectively. Validation examples
        are similarly in `self.validX` and `self.validY`, and test
        examples are in `self.testX` and self.testY`.

        You can also access the training, validation, and testing sets
        with the `get_training_examples()`, `get_validation_exmples()`
        and `get_test_examples()` functions.

        The `trainX`, `validX`, and `testX` arrays are of shape:
            `[num_examples, height, width, n_channels]`

        (For DogSet `height` == `width`.)

        The `trainY`, `validY`, and `testY` arrays are of shape:
            `[num_examples]`

        """
        self.path_to_dogs_csv = os.path.join(path_to_dogsset, 'dogs.csv')
        self.images_dir = os.path.join(path_to_dogsset, 'images')
        np.random.seed(0)
        self.num_classes = num_classes
        # Load in all the data we need from disk
        self.metadata = pd.read_csv(self.path_to_dogs_csv)
        self.semantic_labels = dict(zip(
            self.metadata['numeric_label'],
            self.metadata['semantic_label']
        ))

        self.trainX, self.trainY = self._load_data('train')
        self.validX, self.validY = self._load_data('valid')
        self.testX, self.testY = self._load_data('test')
        self.all_index = np.arange(len(self.trainX) + len(self.testX))
        self.all_count = 0
        self.valid_count = 0

    def get_train_examples(self):
        """
        Gets all training examples in DogSet.
        :return:
            (np.ndarray, np.ndarray), all training examples and all training labels
        """
        return self.trainX, self.trainY

    def get_validation_examples(self):
        """
        Gets all validation examples in DogSet.
        :return:
            (np.ndarray, np.ndarray), all validation examples and all validation labels
        """
        return self.validX, self.validY

    def get_test_examples(self):
        """
        Gets all test examples in DogSet.
        :return:
            (np.ndarray, np.ndarray), all test examples and all test labels
        """
        return self.testX, self.testY

    def get_examples_by_label(self, partition, label, num_examples=None):
        """
        Returns the entire subset of the partition that belongs to the class
        specified by label. If num_examples is None, returns all relevant
        examples.
        """
        if partition == 'train':
            X = self.trainX[self.trainY == label]
        elif partition == 'valid':
            X = self.validX[self.validY == label]
        elif partition == 'test':
            X = self.testX[self.testY == label]
        else:
            raise ValueError('Partition {} does not exist'.format(partition))
        return X if num_examples == None else X[:num_examples]

    def get_semantic_label(self, numeric_label):
        """
        Returns the string representation of the numeric class label (e.g.,
        the numberic label 1 maps to the semantic label 'miniature_poodle').
        """
        return self.semantic_labels[numeric_label]

    def _load_data(self, partition='train'):
        """
        Loads a single data partition from file.
        """
        print("loading %s..." % partition)
        Y = None
        if partition == 'all':
            X = self._get_images(
                self.metadata[~self.metadata.partition.isin(['train', 'valid', 'test'])])
            X = self._preprocess(X, False)
            return X
        else:
            X, Y = self._get_images_and_labels(
                self.metadata[self.metadata.partition == partition])
            X = self._preprocess(X, True)
            return X, Y

    def _get_images_and_labels(self, df):
        """
        Fetches the data based on image filenames specified in df.
        If training is true, also loads the labels.
        """
        X, y = [], []
        for i, row in df.iterrows():
            label = row['numeric_label']
            if label >= self.num_classes: continue
            image = imread(os.path.join(self.images_dir, row['filename']))
            X.append(image)
            y.append(row['numeric_label'])
        return np.array(X), np.array(y).astype(int)

    def _get_images(self, df):
        X = []
        for i, row in df.iterrows():
            image = imread(os.path.join(self.images_dir, row['filename']))
            X.append(image)
        return np.array(X)

    def _preprocess(self, X, is_train):
        """
        Preprocesses the data partition X by normalizing the images
        """
        X = self._normalize(X, is_train)
        return X

    def _normalize(self, X, is_train):
        """
        Normalizes the partition to have mean 0 and variance 1. Learns the
        mean and standard deviation parameters from the training set and
        applies these values when normalizing the other data partitions.

        Returns:
            the normalized data as a numpy array.
        """
        if is_train:
            self.image_mean = np.mean(X, axis=(0,1,2))
            self.image_std = np.std(X, axis=(0,1,2))
        return (X - self.image_mean) / self.image_std

