from typing import List
import numpy as np
import pandas as pd

from sklearn import ensemble, linear_model
import torch
import torch.nn as nn

# from torch import feature_alpha_dropout
from torch.utils.data import DataLoader
import xgboost as xgb

from common.cat_encoding import OneHotEncoder, TruncatedSVD, LabelEncoder
from models.embeddings import EmbeddingsNN, CategoricalDataset

from tqdm import tqdm


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


class EmbeddingsModel(ModelInterface):
    def __init__(self, train: pd.DataFrame, valid: pd.DataFrame, features: List[str]):
        self.df_train = train
        self.df_valid = valid
        self.features = features

        self.epochs = 3
        self.batch_size = 1024

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def encode(self):
        self.x_train, self.x_valid = LabelEncoder(
            self.df_train, self.df_valid, self.features
        )

    def fit(self) -> pd.DataFrame:
        data = pd.concat([self.df_train, self.df_valid], ignore_index=True)
        self.model = EmbeddingsNN(data, self.features).to(self.device)

        # Dataset
        train_ds = CategoricalDataset(
            self.x_train, self.df_train.target.values, self.device
        )

        # Dataloader (dataset + batch size + shuffle)
        params = {"batch_size": self.batch_size, "shuffle": True}
        train_dl = DataLoader(train_ds, **params)

        # using Adam optimizer, can add lr=, weight_decay=
        opt = torch.optim.Adam(self.model.parameters())

        criterion = nn.BCELoss()

        num_batch = len(train_dl)
        for epoch in tqdm(range(self.epochs)):
            t = tqdm(iter(train_dl), leave=False, total=num_batch)
            for x, y in t:
                t.set_description(f"Epoch {epoch}")

                opt.zero_grad()  # find where the grads are zero
                pred = self.model(x)
                loss = criterion(pred[:, 1], y)

                loss.backward()  # do backprop
                opt.step()
                # scheduler.step()

                t.set_postfix(loss=loss.item())

    def predict(self) -> pd.DataFrame:
        # predict on validation data

        # Dataset
        valid_ds = CategoricalDataset(
            self.x_valid, self.df_valid.target.values, self.device
        )

        # Dataloader
        params = {"batch_size": self.batch_size, "shuffle": False}
        valid_dl = DataLoader(valid_ds, **params)

        preds = list()
        for x, y in tqdm(valid_dl, leave=False):
            pred = self.model(x)

            preds += list(pred.cpu().data.numpy())

        # we need the probability values as we are calculating AUC
        # we will use the probability of 1s
        return np.array(preds)[:, 1]
