import nltk
from . import model, abstract

try:
    nltk.corpus.stopwords.words()
except OSError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')