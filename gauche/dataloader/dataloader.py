"""
Abstract class implementing the data loading, data splitting,
type validation and feature extraction functionalities.
"""

from abc import ABCMeta, abstractmethod

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataLoader(metaclass=ABCMeta):
    def __init__(self):
        self.task = None

    @property
    @abstractmethod
    def features(self):
        raise NotImplementedError

    @features.setter
    @abstractmethod
    def features(self, value):
        raise NotImplementedError

    @property
    @abstractmethod
    def labels(self):
        raise NotImplementedError

    @labels.setter
    @abstractmethod
    def labels(self, value):
        raise NotImplementedError

    @abstractmethod
    def validate(self, drop=True):
        """Checks whether the loaded data is a valid instance of the specified
        data type, potentially dropping invalid entries.

        :param drop:  whether to drop invalid entries
        :type drop: bool
        """
        raise NotImplementedError

    @abstractmethod
    def featurize(self, representation):
        """Transforms the features to the specified representation (in-place).

        :param representation: desired feature format
        :type representation: str

        """
        raise NotImplementedError

    def split_and_scale(
        self, test_size: float = 0.2, scale_labels: bool = True, scale_features: bool = False
    ):
        """Splits the data into training and testing sets.

        :param test_size: the relative size of the test set
        :type test_size: float
        :param scale_labels: whether to standardize the labels (after splitting)
        :type scale_labels: bool
        :param scale_features: whether to standardize the features (after splitting)
        :type scale_features: bool
        :returns: (potentially standardized) training and testing sets with associated scalers
        """

        # reshape labels
        self.labels = self.labels.reshape(-1, 1)

        # auxiliary function to perform scaling
        def scale(train, test):
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train)
            test_scaled = scaler.transform(test)
            return train_scaled, test_scaled, scaler

        # split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=test_size, random_state=1
        )

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

        # return concatenated tuples
        return features_out + labels_out
