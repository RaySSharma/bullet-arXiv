import re
import string

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Abstract:
    """Class for basic transformation of paper abstracts. Allows for stripping LaTeX, digits, punctuation.
    """

    def __init__(self, text):
        """Constructor for Abstract

        Args:
            text (str): Raw paper abstract text.
        """
        self.text = text

    def _clean_garbage(self):
        garbage = '\ufeff|â€™|â€"|â€œ|â€˜|â€\x9d|â€œi|_|â€|£|éé|ô|à|â|ê|—£|éè|ü|é|œ|î|æ|ç|‘|é—|…|ö|è'

        cleaned = re.sub(garbage, "", self.text)
        cleaned = cleaned.replace("-", " ")
        cleaned = re.sub(r"\n+", " ", cleaned)
        cleaned = re.sub(r"\'", "'", cleaned)
        cleaned = cleaned.replace("-", " ").replace("—", " ")
        self.text = cleaned
        return self

    def _clean_latex(self):
        self.text = re.sub(r"\$.*?\$", "", self.text)
        self.text = re.sub(r"\{.*?\}", "", self.text)
        return self

    def _clean_digits(self):
        self.text = re.sub(r"\d+", "", self.text)
        return self

    def _clean_punctuation(self):
        pattern = r"[" + string.punctuation + "]"
        self.text = re.sub(pattern, "", self.text)
        return self

    def clean(self):
        return self._clean_latex()._clean_digits()._clean_punctuation()


class FormatText(BaseEstimator, TransformerMixin):
    """Format text by casting as Abstract and cleaning.
    """

    def __init__(self,):
        super().__init__()

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X_ = X["abstract"]
        else:
            X_ = X

        X_ = X_.apply(lambda x: Abstract(x).clean().text).str.lower()
        return X_

    def fit(self, X, y=None):
        return self

