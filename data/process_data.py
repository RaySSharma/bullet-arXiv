"""Separates data scraped by arxivabscraper, splitting into training/validation/test sets.
"""
import pathlib

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 0
RAW_DATA = pathlib.Path("../data/raw/arxiv_abstracts.csv")
PROC_DIR = pathlib.Path("../data/processed/")


def train_test_val_split(x_input):
    """Train/validation/test set split.

    Args:
        x (array-like): Underlying data to split.

    Returns:
        train_train (array-like): Training data.
        test (array-like): Testing data.
        train_val (array-like): Validation data.
    """
    train, test = train_test_split(
        x_input, test_size=0.2, shuffle=True, random_state=RANDOM_STATE
    )
    train_train, train_val = train_test_split(
        train, test_size=0.25, shuffle=True, random_state=RANDOM_STATE
    )
    return train_train, test, train_val


if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA)
    df = df.loc[~df.doi.isna()]  # Require DOI numbers for papers.

    X_train, X_test, X_val = train_test_val_split(df)

    X_test.to_csv(PROC_DIR.joinpath("test.csv"), index=False)
    X_train.to_csv(PROC_DIR.joinpath("train.csv"), index=False)
    X_val.to_csv(PROC_DIR.joinpath("val.csv"), index=False)
