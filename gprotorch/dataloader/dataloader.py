"""
Abstract class implementing the data loading, data splitting
and feature extraction functionalities.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from abc import ABCMeta, abstractmethod


class DataLoader(metaclass=ABCMeta):

    def __init__(self, task):
        self.task = task
        self.features = None
        self.labels = None

    @property
    @abstractmethod
    def data(self):
        raise NotImplementedError

    @data.setter
    @abstractmethod
    def data(self, dataset):
        raise NotImplementedError

    @abstractmethod
    def validate(self, drop=True):
        """Checks whether the loaded data is a valid instance of the specified
        data type, potentially dropping invalid entries (in-place).

        Args:
            drop (bool):  whether to drop invalid entries (in-place)

        """
        raise NotImplementedError

    @abstractmethod
    def featurize(self, representation):
        """Transforms the features to the specified representation (in-place).

        Args:
            representation: desired feature format

        """
        raise NotImplementedError

    def split_and_scale(self, test_size=.2, scale_labels=True, scale_features=False):
        """Splits the data into training and testing sets.

        Args:
            test_size: the relative size of the test set
            scale_labels: whether to standardize the labels (after splitting)
            scale_features: whether to standardize the features (after splitting)

        Returns:
            (potentially standardized) training and testing sets with associated scalers

        """

        # auxiliary function to perform scaling
        def scale(train, test):
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train)
            test_scaled = scaler.transform(test)
            return train_scaled, test_scaled, scaler

        # split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=test_size, random_state=1)

        # scale features, if requested
        if scale_features:
            features_out = scale(X_train, X_test)
        else:
            features_out = X_train, X_test, None

        # scale labels, if requested
        if scale_labels:
            labels_out = scale(y_train, y_test)
        else:
            labels_out = y_train, y_test, None

        return features_out, labels_out
