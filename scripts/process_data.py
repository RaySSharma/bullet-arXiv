"""Separates data scraped by arxivscraper, splitting into training/validation/test sets.
"""
import argparse
import pathlib

import pandas as pd
from sklearn.model_selection import train_test_split
from src import ROOT_DIR

RANDOM_STATE = 0
PROC_DIR = pathlib.Path("data/processed/")


def train_test_val_split(x_input, test_split=0.2, val_split=0.2):
    """Train/validation/test set split.

    Args:
        x (array-like): Underlying data to split.

    Returns:
        train_train (array-like): Training data.
        test (array-like): Testing data.
        train_val (array-like): Validation data.
    """
    val_split_partial = 1 - val_split / (1 - test_split)
    train, test = train_test_split(
        x_input, test_size=test_split, shuffle=True, random_state=RANDOM_STATE
    )
    train_train, train_val = train_test_split(
        train, test_size=val_split_partial, shuffle=True, random_state=RANDOM_STATE
    )
    return train_train, test, train_val


def parse_args():
    parser = argparse.ArgumentParser(description="Args")
    parser.add_argument(
        "test_split", type=float, help="Fraction of all data to hold out for testing."
    )
    parser.add_argument(
        "val_split", type=float, help="Fraction of all data to hold out for validation."
    )
    parser.add_argument("infile", type=pathlib.Path, help="Path to raw CSV file.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    test_split = args.test_split
    val_split = args.val_split
    infile = args.infile

    df = pd.read_csv(infile)
    df = df.loc[~df.doi.isna()]  # Require DOI numbers for papers.

    X_train, X_test, X_val = train_test_val_split(df, test_split, val_split)

    X_test.to_csv(ROOT_DIR / "data/processed/test.csv", index=False)
    X_train.to_csv(ROOT_DIR / "/data/processed/train.csv", index=False)
    X_val.to_csv(ROOT_DIR / "data/processed/val.csv", index=False)
