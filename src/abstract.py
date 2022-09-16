import re
import string


class Abstract:
    """Class for basic transformation of paper abstracts. Allows for stripping LaTeX, digits, punctuation.
    """

    def __init__(self, text):
        """Constructor for Abstract

        Args:
            text (str): Raw paper abstract text.
        """
        self.text = text

    def strip_latex(self):
        self.text = re.sub(r"\$.*?\$", "", self.text)
        return self

    def strip_digits(self):
        self.text = re.sub(r"\d+", "", self.text)
        return self

    def strip_punctuation(self):
        pattern = r"[" + string.punctuation + "]"
        self.text = re.sub(pattern, "", self.text)
        return self

    def clean(self):
        return self.strip_latex().strip_digits().strip_punctuation()

