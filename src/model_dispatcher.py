from typing import List
import pandas as pd

from sklearn import ensemble, linear_model
import xgboost as xgb

from common.cat_encoding import OneHotEncoder, TruncatedSVD, LabelEncoder


class ModelInterface:
    def encode(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass


class LogisticRegressionModel(ModelInterface):
    def __init__(self, train: pd.DataFrame, valid: pd.DataFrame, features: List[str]):
        self.df_train = train
        self.df_valid = valid
        self.features = features

    def encode(self):
        self.x_train, self.x_valid = OneHotEncoder(
            self.df_train, self.df_valid, self.features
        )

    def fit(self) -> pd.DataFrame:
        self.model = linear_model.LogisticRegression()

        # fit model on training data
        self.model.fit(self.x_train, self.df_train.target.values)

    def predict(self) -> pd.DataFrame:
        # predict on validation data
        # we need the probability values as we are calculating AUC
        # we will use the probability of 1s
        return self.model.predict_proba(self.x_valid)[:, 1]


class DecisionTreeModel(ModelInterface):
    def __init__(self, train: pd.DataFrame, valid: pd.DataFrame, features: List[str]):
        self.df_train = train
        self.df_valid = valid
        self.features = features

    def encode(self):
        self.x_train, self.x_valid = LabelEncoder(
            self.df_train, self.df_valid, self.features
        )

    def fit(self) -> pd.DataFrame:
        self.model = ensemble.RandomForestClassifier(n_jobs=-1)

        # fit model on training data
        self.model.fit(self.x_train, self.df_train.target.values)

    def predict(self) -> pd.DataFrame:
        # predict on validation data
        # we need the probability values as we are calculating AUC
        # we will use the probability of 1s
        return self.model.predict_proba(self.x_valid)[:, 1]


class DecisionTreeModelSVD(DecisionTreeModel):
    def enconde(self):
        x_train, x_valid = OneHotEncoder(self.df_train, self.df_valid, self.features)
        self.x_train, self.x_valid = TruncatedSVD(x_train, x_valid, 120)


class XGBoost(DecisionTreeModel):
    def fit(self) -> pd.DataFrame:
        self.model = xgb.XGBClassifier(
            n_jobs=-1, max_depth=7, n_estimators=200, verbosity=0
        )

        # fit model on training data
        self.model.fit(self.x_train, self.df_train.target.values)
