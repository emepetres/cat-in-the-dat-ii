from typing import List
import pandas as pd

from sklearn import ensemble, linear_model, preprocessing


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
        df_train = self.df_train
        df_valid = self.df_valid
        features = self.features

        # initialize OneHotEncoder from scikit-learn
        ohe = preprocessing.OneHotEncoder()

        # fit ohe on training + validation features
        # (do this way as it would be with training + testing data)
        full_data = pd.concat([df_train[features], df_valid[features]], axis=0)
        ohe.fit(full_data[features])

        # transform training data
        self.x_train = ohe.transform(df_train[features])

        # transform validation data
        self.x_valid = ohe.transform(df_valid[features])

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
        df_train = self.df_train
        df_valid = self.df_valid
        features = self.features

        # fit ohe on training + validation features
        # (do this way as it would be with training + testing data)
        full_data = pd.concat([df_train[features], df_valid[features]], axis=0)

        for col in features:
            lbl = preprocessing.LabelEncoder()

            # fit the label encoder on all data
            lbl.fit(full_data[col])

            # transform all the data
            df_train.loc[:, col] = lbl.transform(df_train[col])
            df_valid.loc[:, col] = lbl.transform(df_valid[col])

        # transform training data
        self.x_train = df_train[features].values

        # transform validation data
        self.x_valid = df_valid[features].values

    def fit(self) -> pd.DataFrame:
        self.model = ensemble.RandomForestClassifier(n_jobs=-1)

        # fit model on training data
        self.model.fit(self.x_train, self.df_train.target.values)

    def predict(self) -> pd.DataFrame:
        # predict on validation data
        # we need the probability values as we are calculating AUC
        # we will use the probability of 1s
        return self.model.predict_proba(self.x_valid)[:, 1]
