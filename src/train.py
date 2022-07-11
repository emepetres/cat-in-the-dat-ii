import argparse
import pandas as pd

from sklearn import metrics

import config
from model_dispatcher import ModelInterface, LogisticRegressionModel, DecisionTreeModel


def run(fold: int, model: ModelInterface):
    # load the full training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    # all columns are features except id, target and kfold columns
    features = [f for f in df.columns if f not in ("id", "target", "kfold")]

    # fill all NaN values with NONE
    # note that I am converting all columns to "strings"
    # it doesn't matter because all are categories
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # initialize Logistic Regression model
    lr_model = model(df_train, df_valid, features)

    # one hot encode of all features (they are all categorical)
    lr_model.encode()

    # fit model on training data
    lr_model.fit()

    # predict on validation data
    valid_preds = lr_model.predict()

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    # print auc
    print(f"Fold = {fold}, AUC = {auc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="lr")

    args = parser.parse_args()

    model = None
    if (args.model == "lr"):
        model = LogisticRegressionModel
    elif (args.model == "rf"):
        model = DecisionTreeModel
    else:
        raise argparse.ArgumentError(
            "Only 'lr' (logistic regression) and 'rf'"
            " (random forest) models are supported")

    for fold_ in range(5):
        run(fold_, model)
