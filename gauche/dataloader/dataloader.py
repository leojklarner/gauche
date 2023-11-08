"""
Abstract class implementing the data loading, data splitting,
type validation and feature extraction functionalities.
"""

from abc import ABCMeta, abstractmethod
from typing import Optional


class DataLoader(metaclass=ABCMeta):
    def __init__(self):
        self.task = None

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = value

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value

    @abstractmethod
    def validate(
        self, drop: Optional[bool] = True, canonicalize: Optional[bool] = True
    ):
        """Checks whether the loaded data is a valid instance
        of the specified data type, optionally dropping invalid
        entries and standardizing the remaining ones.

        :param drop:  whether to drop invalid entries
        :type drop: bool
        :param canonicalize: whether to standardize the data
        :type canonicalize: bool
        """
        raise NotImplementedError

    @abstractmethod
    def featurize(self, representation: str, **kwargs):
        """Transforms the features to the specified representation (in-place).

        :param representation: desired feature format
        :type representation: str
        :param kwargs: additional keyword arguments for the representation function
        :type kwargs: dict
        """
        raise NotImplementedError
