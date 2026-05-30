# Cleans and prepares job descriptions for text analysis.

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data on first use
def _ensure_nltk_data():
    for resource in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk_data()

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))

# Common filler words that appear in job postings but carry no skill signal
_EXTRA_STOPWORDS = {
    "experience", "work", "working", "including", "strong", "ability",
    "knowledge", "skills", "skill", "years", "year", "team", "role",
    "position", "job", "candidate", "opportunity", "responsibilities",
    "requirements", "preferred", "required", "must", "plus", "well",
    "also", "within", "across", "new", "used", "use", "using",
    "good", "great", "excellent", "related", "field", "minimum",
    "least", "equivalent", "desired", "highly", "seeking", "looking",
    "join", "help", "provide", "ensure", "maintain", "support", "us",
}


def clean_text(text: str) -> str:
    """Lowercase, remove HTML tags, punctuation, digits, and extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # strip URLs
    text = re.sub(r"\d+", " ", text)               # remove digits
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Split cleaned text into word tokens."""
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove NLTK stopwords plus domain-specific filler words."""
    combined = _stop_words | _EXTRA_STOPWORDS
    return [t for t in tokens if t not in combined and len(t) > 2]


def lemmatize(tokens: list[str]) -> list[str]:
    """Lemmatize a list of tokens."""
    return [_lemmatizer.lemmatize(t) for t in tokens]


def preprocess_text(text: str) -> list[str]:
    """Full pipeline: clean → tokenize → remove stopwords → lemmatize."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return tokens


def preprocess_series(series):
    """
    Apply the full preprocessing pipeline to a pandas Series of raw text.

    Returns a new Series where each entry is a list of clean tokens.
    """
    return series.apply(preprocess_text)


def tokens_to_string(tokens: list[str]) -> str:
    """Rejoin tokens into a single string (useful for TF-IDF vectorizers)."""
    return " ".join(tokens)