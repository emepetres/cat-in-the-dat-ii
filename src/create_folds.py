import pandas as pd
from sklearn import model_selection

from common.kaggle import download_competition_data
import config

if __name__ == "__main__":
    # Download competition data if necessary
    download_competition_data(config.COMPETITION, config.DATA_INPUT)

    # Read training data
    df = pd.read_csv(config.TRAINING_DATA)

    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch labels
    y = df.target.values

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f

    # save the new csv with kfold column
    df.to_csv(config.TRAINING_FILE, index=False)
