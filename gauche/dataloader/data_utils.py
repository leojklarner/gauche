"""
Utility functions for molecular data
"""

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def transform_data(
    X_train: np.array,
    y_train: np.array,
    X_test: np.array,
    y_test: np.array,
    use_pca: Optional[bool] = False,
    n_components: Optional[int] = 10,
) -> tuple:
    """
    Apply feature scaling, dimensionality reduction to the data. Return the standardised and low-dimensional train and
    test sets together with the scaler object for the target values.

    :param X_train: training set features
    :type X_train: np.array
    :param y_train: training set targets
    :type y_train: np.array
    :param X_test: test set features
    :type X_test: np.array
    :param y_test: test set targets
    :type y_test: np.array
    :param use_pca: whether to use PCA for dimensionality reduction
    :type use_pca: bool
    :param n_components: number of principal components to retain
    :type n_components: int
    :return: X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_scaler
    """

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    if use_pca:
        pca = PCA(n_components)
        X_train_scaled = pca.fit_transform(X_train)
        print(
            "Fraction of variance retained is: "
            + str(sum(pca.explained_variance_ratio_))
        )
        X_test_scaled = pca.transform(X_test)

    return (
        X_train_scaled,
        y_train_scaled,
        X_test_scaled,
        y_test_scaled,
        y_scaler,
    )
