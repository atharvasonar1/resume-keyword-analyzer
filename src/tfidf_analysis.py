# Identifies role-specific keywords using TF-IDF.

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocessing import preprocess_series, tokens_to_string


def build_tfidf_matrix(corpus: list,
                        max_features: int = 5000,
                        ngram_range: tuple = (1, 2)):
    """
    Fit a TF-IDF vectorizer on a list of preprocessed document strings.

    Parameters
    ----------
    corpus       : List of strings (one per job description, already cleaned).
    max_features : Maximum vocabulary size.
    ngram_range  : Tuple (min_n, max_n) for n-gram extraction.

    Returns
    -------
    vectorizer   : Fitted TfidfVectorizer.
    tfidf_matrix : Sparse matrix of shape (n_docs, n_features).
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,  # apply log normalization to term frequency
        min_df=1,           # keep all terms; safe for small filtered datasets
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix


def get_top_tfidf_terms(vectorizer: TfidfVectorizer,
                         tfidf_matrix,
                         top_n: int = 30) -> pd.DataFrame:
    """
    Rank terms by their mean TF-IDF score across all documents.

    Parameters
    ----------
    vectorizer   : Fitted TfidfVectorizer.
    tfidf_matrix : Sparse matrix returned by build_tfidf_matrix.
    top_n        : Number of top terms to return.

    Returns
    -------
    DataFrame with columns ['term', 'mean_tfidf'].
    """
    feature_names = vectorizer.get_feature_names_out()
    mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

    sorted_indices = np.argsort(mean_scores)[::-1][:top_n]
    rows = [
        {"term": feature_names[i], "mean_tfidf": round(float(mean_scores[i]), 6)}
        for i in sorted_indices
    ]
    return pd.DataFrame(rows)


def get_tfidf_by_document(vectorizer: TfidfVectorizer,
                           tfidf_matrix,
                           top_n: int = 10) -> list:
    """
    For each document return its top-N TF-IDF terms.

    Useful for inspecting which keywords characterise individual postings.

    Returns
    -------
    List of DataFrames (one per document), each with columns ['term', 'tfidf'].
    """
    feature_names = vectorizer.get_feature_names_out()
    results = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[i].toarray().flatten()
        top_indices = np.argsort(row)[::-1][:top_n]
        df = pd.DataFrame({
            "term": feature_names[top_indices],
            "tfidf": row[top_indices].round(6),
        })
        results.append(df)
    return results


def run_tfidf_analysis(filtered_df: pd.DataFrame,
                        description_col: str = "description",
                        top_n: int = 30,
                        max_features: int = 5000,
                        ngram_range: tuple = (1, 2)) -> tuple:
    """
    Full TF-IDF pipeline: preprocess -> vectorize -> rank terms.

    Parameters
    ----------
    filtered_df     : DataFrame of job postings already filtered by role.
    description_col : Column containing raw job description text.
    top_n           : Number of top TF-IDF terms to surface.
    max_features    : Vocabulary cap for the vectorizer.
    ngram_range     : N-gram range for the vectorizer.

    Returns
    -------
    (top_terms_df, vectorizer, tfidf_matrix)
        top_terms_df : DataFrame with the highest-scoring terms.
        vectorizer   : Fitted TfidfVectorizer (reusable for transforms).
        tfidf_matrix : Sparse matrix (useful for further analysis).
    """
    if filtered_df.empty:
        print("Warning: empty DataFrame passed to run_tfidf_analysis.")
        return pd.DataFrame(columns=["term", "mean_tfidf"]), None, None

    if description_col not in filtered_df.columns:
        raise ValueError(f"Column '{description_col}' not found in DataFrame.")

    print(f"Running TF-IDF on {len(filtered_df)} job descriptions...")

    # Preprocess to tokens, then rejoin as strings for the vectorizer
    token_series = preprocess_series(filtered_df[description_col])
    corpus = token_series.apply(tokens_to_string).tolist()

    vectorizer, tfidf_matrix = build_tfidf_matrix(
        corpus,
        max_features=max_features,
        ngram_range=ngram_range,
    )

    top_terms_df = get_top_tfidf_terms(vectorizer, tfidf_matrix, top_n=top_n)

    print(f"TF-IDF complete. Vocabulary size: {len(vectorizer.vocabulary_)}")
    return top_terms_df, vectorizer, tfidf_matrix