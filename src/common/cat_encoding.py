import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn import preprocessing, decomposition
from scipy import sparse


def OneHotEncoder(
    df_train: pd.DataFrame, df_valid: pd.DataFrame, features: List[str]
) -> Tuple[np.ndarray | sparse.csr_matrix, np.ndarray | sparse.csr_matrix]:
    """Best sparse optimization, but slow on trees algorithms"""
    # initialize OneHotEncoder from scikit-learn
    ohe = preprocessing.OneHotEncoder()

    # fit ohe on training + validation features
    # (do this way as it would be with training + testing data)
    full_data = pd.concat([df_train[features], df_valid[features]], axis=0)
    ohe.fit(full_data[features])

    # transform training & validation data
    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    # return training & validation features
    return (x_train, x_valid)


def TruncatedSVD(
    x_train: sparse.csr_matrix, x_valid: sparse.csr_matrix, n_components: int
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Used over a OneHotEnconding to reduce its size"""
    # initialize TruncatedSVD
    # we are reducing the data to 120 components
    svd = decomposition.TruncatedSVD(n_components=n_components)

    # fit svd on full sparse training data
    full_sparse = sparse.vstack((x_train, x_valid))
    svd.fit(full_sparse)

    # transform sparse training data
    x_train = svd.transform(x_train)

    # transform sparse valid data
    x_valid = svd.transform(x_valid)

    return (x_train, x_valid)


def LabelEncoder(
    df_train: pd.DataFrame, df_valid: pd.DataFrame, features: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Used for tree-based algorithms"""
    # fit LabelEncoder on training + validation features
    # (do this way as it would be with training + testing data)
    full_data = pd.concat([df_train[features], df_valid[features]], axis=0)

    for col in features:
        # initialize LabelEncoder for each feature column
        lbl = preprocessing.LabelEncoder()

        # fit the label encoder on all data
        lbl.fit(full_data[col])

        # transform all the data
        df_train.loc[:, col] = lbl.transform(df_train[col])
        df_valid.loc[:, col] = lbl.transform(df_valid[col])

    # return training & validation features
    return (df_train[features].values, df_valid[features].values)
