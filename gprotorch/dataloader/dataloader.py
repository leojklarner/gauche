"""
Abstract base class implementing the data loading, data splitting,
type validation and feature extraction functionalities.

Author: Leo Klarner (https://github.com/leojklarner), March 2021
"""

from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import (KFold, ShuffleSplit, StratifiedKFold,
                                     StratifiedShuffleSplit)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import type_of_target


class DataLoader(metaclass=ABCMeta):
    """
    Abstract DataLoader base class to provide the framework for storing
    the task-specific internal representation of the molecules/complexes in question (self.objects),
    deriving a variety of featurisations (self.features) and the corresponding labels (self.labels)
    and splitting them into disjunct sets for measuring model performance and generalisability.
    """

    def __init__(self, validate_internal_rep=True):
        """
        Class initialisation.

        Args:
            validate_internal_rep: whether to validate the internal representations when reading them in

        Returns: None

        """
        self._objects = None
        self._features = None
        self._labels = None
        self.validate_internal_rep = validate_internal_rep

    @property
    def objects(self):
        """
        Internal representation of the objects being loaded (e.g. SMILES, PDB/SDF file paths)

        Returns: Currently loaded internal representation.

        """

        return self._objects

    @objects.setter
    def objects(self, value):
        """
        Property setter for setting internal representation.
        Runs the self._validate method (if specified upon initialisation),
        dropping any invalid entries and storing only the valid ones.

        Args:
            value: internal representations to be set

        Returns: None

        """

        if self.validate_internal_rep:

            # get valid and invalid entries
            valid, invalid = self._validate(value)

            # set valid entries as internal representation
            self._objects = valid

            # print which invalid entries have been dropped
            if invalid:
                print(
                    f"The following {len(invalid)} entries could not be parsed to a valid"
                    f"internal representation and were dropped."
                )
                print(invalid)
        else:
            self._objects = value

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
        Property setter to read in a specific set of features.

        Args:
            value: the features to be loaded

        Returns: None

        """

        self._features = value

    @property
    def labels(self):
        """
        Property for storing a set of labels.

        Returns: the currently loaded labels.

        """

        return self._labels

    @labels.setter
    def labels(self, value):
        """
        Property setter for loading a specific set of labels.

        Args:
            value: the labels to be loaded

        Returns: None

        """

        self._labels = value

    @abstractmethod
    def _validate(self, data):
        """
        A method to check whether the provided data is a valid instance of, and
        and can be parsed to, the task-specific representation.

        Args:
            data: the objects to be checked

        Returns: (valid, invalid) tuple of valid and invalid entries

        """

        raise NotImplementedError

    @abstractmethod
    def featurize(self, representation):
        """
        Applies a transformation to the internal representation of the loaded objects,
        calculating and storing a set of desired features.

        Args:
            representation: a string or list of strings specifying which transformations should be applied

        Returns: None

        """
        raise NotImplementedError

    @staticmethod
    def _scale(train, test=None):
        """
        Auxiliary function to perform scaling on features and labels.
        Fits the standardisation scaler on the training set and
        transforms training and (optionally) test set

        Args:
            train: numpy array of features/labels used for training
            test: numpy array of features/labels used for testing (optional)

        Returns: a tuple of scaled training set, scaled test set and the fitted scaler

        """

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train)
        test_scaled = scaler.transform(test)

        return train_scaled, test_scaled, scaler

    def split_and_scale(
        self,
        kfold_shuffle=True,
        num_splits=5,
        test_size=0.2,
        scale_labels=True,
        label_dim=True,
        scale_features=False,
    ):

        """
        Method to convert the features and labels to numpy arrays (if necessary) and
        split them into training, validation and test sets according to the specified method.

        Args:
            kfold_shuffle: whether to split the data into k folds (True) or k re-shuffled splits (False)
            num_splits: number of folds or re-shuffled splits
            test_size: size of the test set for re-shuffled splits
            scale_labels: whether to standardize the labels (after splitting)
            label_dim: whether the dimension of the labels should be (n,) or (n,1)
            scale_features: whether to standardize the features (after splitting)

        Returns: a tuple (X_train, X_test, X_scaler, y_train, y_test, y_scaler) of the split
        (and potentially scaled, scalers are None if not) features and labels.

        """

        # check the feature and label datatype, convert them
        # to numpy arrays if necessary

        if isinstance(self.features, pd.DataFrame):
            features = self.features.to_numpy(copy=True)
        elif isinstance(self.features, list):
            features = np.array(self.features)
        else:
            features = self.features

        if isinstance(self.labels, pd.DataFrame):
            labels = self.labels.to_numpy(copy=True)
        elif isinstance(self.labels, list):
            labels = np.array(self.labels)
        else:
            labels = self.labels

        # squeeze or unsqueeze labels if necessary

        if len(labels.shape) > 1 and not label_dim:
            labels = np.squeeze(labels)

        if len(labels.shape) == 1 and label_dim:
            labels = np.expand_dims(labels)

        # check whether the labels are continuous or categorical
        # and select the appropriate (stratified) splitting function

        if type_of_target(labels) == "continuous":

            if kfold_shuffle:
                splitter = KFold(n_splits=num_splits, shuffle=True)

            else:
                splitter = ShuffleSplit(n_splits=num_splits, test_size=test_size)

        else:
            if kfold_shuffle:
                splitter = StratifiedKFold(n_splits=num_splits, shuffle=True)

            else:
                splitter = StratifiedShuffleSplit(
                    n_splits=num_splits, test_size=test_size
                )

        # create and store the splits in accordance to the specified parameters

        splits = []

        for train_index, test_index in splitter.split(features, labels):
            X_train = features[train_index]
            X_test = features[test_index]
            y_train = labels[train_index]
            y_test = labels[test_index]

            # scale features, if required

            if scale_features:
                features_out = self._scale(X_train, X_test)
            else:
                features_out = X_train, X_test, None

            # scale labels, if required

            if scale_labels:
                labels_out = self._scale(y_train, y_test)
            else:
                labels_out = y_train, y_test, None

            splits.append(features_out + labels_out)

        # return list of concatenated tuples for each split

        return splits
