"""
Abstract class implementing the data loading, data splitting,
type validation and feature extraction functionalities.
"""

from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import type_of_target
import pandas as pd


class DataLoader(metaclass=ABCMeta):
    """
    Abstract class implementing the data loading, data splitting,
    type validation and feature extraction functionalities.
    """

    def __init__(self):
        self._objects = None
        self._features = None
        self._labels = None

    @property
    def objects(self):
        """
        Internal representation of the objects being loaded (e.g. SMILES, PDB codes)

        Returns: Currently loaded internal representation.
        """

        return self._objects

    @objects.setter
    def objects(self, value):
        """
        Abstract method for setting internal representation. Runs the validate method over
        each entry and drops invalid ones.

        Args:
            value: internal representations to be set

        Returns: None

        """

        # get valid and invalid entries
        valid, invalid = self._validate(value)

        # set valid entries as internal representation
        self._objects = valid

        # print which invalid entries have been dropped
        if invalid:
            print(f"The entries")
            for i in invalid:
                print(i)
            print(f"have been dropped, as they could not be parsed into valid representations.")

    @property
    def features(self):
        """
        Property for storing features.

        Returns: currently loaded features.
        """
        return self._features

    @features.setter
    def features(self, value):
        """
        An method to load a specific set of features.

        Args:
            value: the features to be loaded

        Returns: None

        """

        self._features = value

    @property
    def labels(self):
        """
        Property for storing labels

        Returns: the currently loaded labels.

        """

        return self._labels

    @labels.setter
    def labels(self, value):
        """
        A method for loading a specific set of labels.

        Args:
            value: the labels to be loaded

        Returns: None

        """

        self._labels = value

    @abstractmethod
    def _validate(self, data):
        """Check whether the provided data is a valid instance of the
        task-specific representation

        Args:
            data: the data to be checked

        Returns: (valid, invalid) tuple of valid and invalid entries

        """

        raise NotImplementedError

    @abstractmethod
    def featurize(self, representation):
        """Transforms the features to the specified representation (in-place).

        Args:
            representation: desired feature format

        """
        raise NotImplementedError

    @staticmethod
    def _scale(train, test=None):
        """
        Auxiliary function to perform scaling on features and labels.
        Fits the standardisation scaler on the training set and
        transforms training and (optionally) test set
        Args:
            train: proportion of features/labels used for training
            test: proportion of features/labels used for testing, optional
        Returns: a tuple of scaled training set, scaled test set and the fitted scaler
        """

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train)
        test_scaled = scaler.transform(test)

        return train_scaled, test_scaled, scaler

    def split_and_scale(self, kfold_shuffle=True, num_splits=5, test_size=0.2, scale_labels=True, scale_features=False):
        """Splits the data into training and test sets.
        Args:
            kfold_shuffle: whether to split the data into k folds (True) or k re-shuffled splits (False)
            num_splits: number of folds or re-shuffled splits
            test_size: size of the test set for re-shuffled splits
            scale_labels: whether to standardize the labels (after splitting)
            scale_features: whether to standardize the features (after splitting)
        Returns:
            (potentially standardized) training and testing sets with associated scalers
        """

        if isinstance(self.labels, pd.DataFrame):
            labels = self.labels.to_numpy()
        else:
            labels = self.labels

        if isinstance(self.features, pd.DataFrame):
            features = self.features.numpy()
        else:
            features = self.features

        # reshape labels
        labels = labels.reshape(-1, 1)

        # use non-stratified methods if labels are continuous
        if type_of_target(labels) == 'continuous':

            if kfold_shuffle:
                splitter = KFold(
                    n_splits=num_splits, shuffle=True
                )

            else:
                splitter = ShuffleSplit(
                    n_splits=num_splits, test_size=test_size
                )

        # use stratified methods if labels are discrete
        else:
            if kfold_shuffle:
                splitter = StratifiedKFold(
                    n_splits=num_splits, shuffle=True
                )

            else:
                splitter = StratifiedShuffleSplit(
                    n_splits=num_splits, test_size=test_size
                )

        splits = []

        # convert features from SMILES list into numpy array
        self.features = np.array(features)

        for train_index, test_index in splitter.split(features, labels):
            X_train = features[train_index]
            X_test = features[test_index]
            y_train = labels[train_index]
            y_test = labels[test_index]

            # scale features, if requested
            if scale_features:
                features_out = self._scale(X_train, X_test)
            else:
                features_out = X_train, X_test, None

            # scale labels, if requested
            if scale_labels:
                labels_out = self._scale(y_train, y_test)
            else:
                labels_out = y_train, y_test, None

            splits.append(features_out + labels_out)

        # return list of concatenated tuples for each split
        return splits
