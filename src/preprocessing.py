# Cleans and prepares job descriptions for text analysis.

import re
import string

try:
    from nltk.corpus import stopwords as nltk_stopwords
    _stop_words = set(nltk_stopwords.words("english"))
except Exception:
    # Fallback if NLTK stopwords are not available
    _stop_words = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
        "your", "yours", "yourself", "he", "him", "his", "himself", "she",
        "her", "hers", "herself", "it", "its", "itself", "they", "them",
        "their", "theirs", "themselves", "what", "which", "who", "whom",
        "this", "that", "these", "those", "am", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "having", "do", "does",
        "did", "doing", "a", "an", "the", "and", "but", "if", "or",
        "because", "as", "until", "while", "of", "at", "by", "for", "with",
        "about", "against", "between", "into", "through", "during", "before",
        "after", "above", "below", "to", "from", "up", "down", "in", "out",
        "on", "off", "over", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "s",
        "t", "can", "will", "just", "don", "should", "now", "d", "ll",
        "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn",
        "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn",
        "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
    }

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

_ALL_STOPWORDS = _stop_words | _EXTRA_STOPWORDS


def clean_text(text: str) -> str:
    """Lowercase, remove HTML tags, punctuation, digits, and extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)           # strip HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # strip URLs
    text = re.sub(r"\d+", " ", text)                # remove digits
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(tokens: list) -> list:
    """Remove stopwords and very short tokens."""
    return [t for t in tokens if t not in _ALL_STOPWORDS and len(t) > 2]


def preprocess_text(text: str) -> list:
    """Full pipeline: clean -> tokenize -> remove stopwords."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    tokens = cleaned.split()
    tokens = remove_stopwords(tokens)
    return tokens


def preprocess_series(series):
    """
    Apply the full preprocessing pipeline to a pandas Series of raw text.

    Returns a new Series where each entry is a list of clean tokens.
    """
    return series.apply(preprocess_text)


def tokens_to_string(tokens: list) -> str:
    """Rejoin tokens into a single string (useful for TF-IDF vectorizers)."""
    return " ".join(tokens)