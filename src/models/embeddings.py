import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pandas as pd
from typing import List


class EmbeddingsNN(nn.Module):
    def __init__(self, df: pd.DataFrame, catcols: List):
        super().__init__()

        # create one embedding per cat variable
        self.embeddings = nn.ModuleList()
        self.concat_emb_features = 0  # register the size of all embeddings concatenated
        for c in catcols:
            # find the number of unique values in the column
            num_unique_values = int(df[c].nunique())
            # simple dimension of embedding calculartor
            # min size is half of the number of unique values
            # max size is 50. it depends on the number of unique categories too
            #   50 is quite sufficient most of the times, but if you have
            #   millions of unique values, you might need a larger dimension
            embed_dim = int(min(np.ceil((num_unique_values / 2)), 50))

            # embedding layer. size is always 1 more than the unique values in input
            emb = nn.Embedding(num_unique_values + 1, embed_dim)

            # 1-d spatial dropout is the standard for embedding layers
            # you can use it in NLP tasks too
            dropout = nn.Dropout1d(0.3)

            self.embeddings.append(nn.Sequential(emb, dropout))
            self.concat_emb_features += embed_dim

        # add a batchnorm layer
        # from here, everything is up to you
        # you can try different architectures, this is the architecture I like to use
        # if you have numerical features, you should add them here or
        # concatenated with the embeddings
        self.batch_norm = nn.BatchNorm1d(self.concat_emb_features)

        # a dense layer with dropout
        # we will repeat it multiple times
        # start with one or two layers only
        hidden_sizes = [300, 300]
        self.dense_layers = nn.Sequential(
            nn.Linear(self.concat_emb_features, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden_sizes[1]),
        )

        # using softmax and treating it as a two class problem
        # you can also use sigmoid, then you need to use only one output class
        # # self.output = nn.Sequential(nn.Linear(300, 2), nn.Softmax())
        self.output = nn.Linear(
            hidden_sizes[1], 2
        )  # not using softmax as we include it in loss

    def forward(self, x):
        embs = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs, dim=1)
        x = self.batch_norm(x)
        x = self.dense_layers(x)
        return self.output(x)


class CategoricalDataset(data.Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, device: str):
        self.x = torch.from_numpy(features).to(device)
        self.y = (
            torch.from_numpy(CategoricalDataset._to_categorical(labels, 2))
            .to(device)
            .to(torch.float32)
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx]]

    def _to_categorical(y: np.ndarray, num_classes: int) -> np.ndarray:
        return np.eye(num_classes, dtype="uint8")[y]
