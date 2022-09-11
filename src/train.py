import argparse
import pandas as pd

from sklearn import metrics

import config
from model_dispatcher import (
    DecisionTreeModelSVD,
    ModelInterface,
    LogisticRegressionModel,
    DecisionTreeModel,
    XGBoost,
)


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

    # initialize model
    custom_model = model(df_train, df_valid, features)

    # encode all features (they are all categorical)
    custom_model.encode()

    # fit model on training data
    custom_model.fit()

    # predict on validation data
    valid_preds = custom_model.predict()

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    # print auc
    print(f"Fold = {fold}, AUC = {auc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="lr")

    args = parser.parse_args()

    model = None
    if args.model == "lr":
        model = LogisticRegressionModel
    elif args.model == "rf":
        model = DecisionTreeModel
    elif args.model == "svd":
        model = DecisionTreeModelSVD
    elif args.model == "xgb":
        model = XGBoost
    else:
        raise argparse.ArgumentError(
            "Only 'lr' (logistic regression)"
            ", 'rf' (random forest)"
            ", 'svd' (random forest with truncate svd)"
            ", 'xgb' (XGBoost)"
            " models are supported"
        )

    for fold_ in range(5):
        run(fold_, model)
