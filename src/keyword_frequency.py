# Calculates common words and skills for a selected role.

from collections import Counter
import pandas as pd
from src.preprocessing import preprocess_series, clean_text, tokens_to_string

# Curated list of technical and soft skills commonly found in job postings.
# Extend this list as needed for your dataset.
# Multi-word skills are supported and matched against the cleaned description string.
SKILL_KEYWORDS = {
    # Programming languages
    "python", "java", "javascript", "typescript", "sql", "r", "scala",
    "kotlin", "swift", "go", "rust", "ruby", "php",
    # Data & ML
    "pandas", "numpy", "scikit", "tensorflow", "pytorch", "keras",
    "spark", "hadoop", "tableau", "powerbi", "excel", "matplotlib",
    "seaborn", "nlp", "machine learning", "deep learning", "regression",
    "classification", "clustering", "forecasting", "statistics",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins",
    "git", "github", "gitlab", "linux", "bash", "airflow", "dbt",
    # Databases
    "mysql", "postgresql", "mongodb", "redis", "snowflake", "bigquery",
    "oracle", "sqlite", "elasticsearch", "cassandra",
    # Web & APIs
    "react", "angular", "vue", "node", "django", "flask", "fastapi",
    "rest", "graphql", "html", "css",
    # Soft skills
    "communication", "leadership", "collaboration", "problem solving",
    "critical thinking", "agile", "scrum", "project management",
}


def get_top_keywords(token_lists: list, top_n: int = 30) -> pd.DataFrame:
    """
    Count the most frequent tokens across all job descriptions.

    Parameters
    ----------
    token_lists : list of token lists produced by preprocessing.
    top_n : number of top keywords to return.

    Returns
    -------
    DataFrame with columns ['keyword', 'count', 'frequency_pct'].
    """
    total_docs = len(token_lists)
    if total_docs == 0:
        return pd.DataFrame(columns=["keyword", "count", "frequency_pct"])

    all_tokens = [token for tokens in token_lists for token in tokens]
    counts = Counter(all_tokens)

    top = counts.most_common(top_n)
    rows = []
    for word, count in top:
        doc_freq = sum(1 for tokens in token_lists if word in tokens)
        rows.append({
            "keyword": word,
            "count": count,
            "frequency_pct": round(doc_freq / total_docs * 100, 2),
        })

    return pd.DataFrame(rows)


def get_skill_frequency(cleaned_texts: list, top_n: int = 30) -> pd.DataFrame:
    """
    Count how often each skill keyword appears across job descriptions.

    Checks against the full cleaned description string so that multi-word
    skills like 'machine learning' are matched correctly.

    Parameters
    ----------
    cleaned_texts : list of cleaned description strings (output of clean_text).
    top_n : number of top skills to return.

    Returns
    -------
    DataFrame with columns ['skill', 'count', 'frequency_pct'].
    """
    total_docs = len(cleaned_texts)
    if total_docs == 0:
        return pd.DataFrame(columns=["skill", "count", "frequency_pct"])

    skill_count = Counter()
    skill_doc_freq = Counter()

    for text in cleaned_texts:
        for skill in SKILL_KEYWORDS:
            if skill in text:
                skill_count[skill] += 1
                skill_doc_freq[skill] += 1

    top = skill_count.most_common(top_n)
    rows = []
    for skill, count in top:
        rows.append({
            "skill": skill,
            "count": count,
            "frequency_pct": round(skill_doc_freq[skill] / total_docs * 100, 2),
        })

    return pd.DataFrame(rows)


def compute_keyword_frequency(filtered_df: pd.DataFrame,
                              description_col: str = "description",
                              top_n: int = 30) -> tuple:
    """
    Main entry point: preprocess descriptions and return both
    overall keyword and skill frequency tables.

    Parameters
    ----------
    filtered_df     : DataFrame of job postings already filtered by role.
    description_col : Name of the column containing job description text.
    top_n           : How many top entries to include in each table.

    Returns
    -------
    (top_keywords_df, top_skills_df)
    """
    if filtered_df.empty:
        print("Warning: empty DataFrame passed to compute_keyword_frequency.")
        empty = pd.DataFrame()
        return empty, empty

    if description_col not in filtered_df.columns:
        raise ValueError(f"Column '{description_col}' not found in DataFrame.")

    print(f"Preprocessing {len(filtered_df)} job descriptions...")

    # Token lists for general keyword frequency
    token_series = preprocess_series(filtered_df[description_col])
    token_lists = token_series.tolist()

    # Cleaned strings for multi-word skill matching
    cleaned_texts = filtered_df[description_col].apply(clean_text).tolist()

    top_keywords_df = get_top_keywords(token_lists, top_n=top_n)
    top_skills_df = get_skill_frequency(cleaned_texts, top_n=top_n)

    return top_keywords_df, top_skills_df