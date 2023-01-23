"""
Utility functions for molecular data.
"""
import numpy as np
from typing import Optional Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def transform_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_components: Optional[int] = None,
    use_pca: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Apply feature scaling, dimensionality reduction to the data.
    
    Returns the standardised and low-dimensional train and
    test sets together with the scaler object for the target values.

    :param X_train: input train data
    :type X_train: np.ndarray
    :param y_train: train labels
    :type y_train: np.ndarray
    :param X_test: input test data
    :type X_test: np.ndarray
    :param y_test: test labels
    :type y_test: np.ndarray
    :param n_components: number of principal components to keep when ``use_pca=True``
    :type n_components: int, optional. Default is ``None``.
    :param use_pca: Whether or not to use PCA.
    :type use_pca: bool
    :return: X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_scaler
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, sklearn.preprocessing.StandardScaler]
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
