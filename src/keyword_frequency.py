# Calculates common words and skills for a selected role.

from collections import Counter
import pandas as pd
from src.preprocessing import preprocess_series, tokens_to_string

# Curated list of technical and soft skills commonly found in job postings.
# Extend this list as needed for your dataset.
SKILL_KEYWORDS = {
    # Programming languages
    "python", "java", "javascript", "typescript", "sql", "r", "scala",
    "kotlin", "swift", "go", "rust", "c", "cpp", "csharp", "ruby", "php",
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


def get_top_keywords(token_lists: list[list[str]], top_n: int = 30) -> pd.DataFrame:
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
    all_tokens = [token for tokens in token_lists for token in tokens]
    counts = Counter(all_tokens)
    total_docs = len(token_lists)

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


def get_skill_frequency(token_lists: list[list[str]], top_n: int = 30) -> pd.DataFrame:
    """
    Filter token counts to only recognised skill keywords.

    Parameters
    ----------
    token_lists : list of token lists produced by preprocessing.
    top_n : number of top skills to return.

    Returns
    -------
    DataFrame with columns ['skill', 'count', 'frequency_pct'].
    """
    total_docs = len(token_lists)
    skill_counter: Counter = Counter()
    skill_doc_freq: Counter = Counter()

    for tokens in token_lists:
        token_set = set(tokens)
        for token in tokens:
            if token in SKILL_KEYWORDS:
                skill_counter[token] += 1
        for skill in SKILL_KEYWORDS:
            if skill in token_set:
                skill_doc_freq[skill] += 1

    top = skill_counter.most_common(top_n)
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
                              top_n: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    if description_col not in filtered_df.columns:
        raise ValueError(f"Column '{description_col}' not found in DataFrame.")

    print(f"Preprocessing {len(filtered_df)} job descriptions...")
    token_series = preprocess_series(filtered_df[description_col])
    token_lists = token_series.tolist()

    top_keywords_df = get_top_keywords(token_lists, top_n=top_n)
    top_skills_df = get_skill_frequency(token_lists, top_n=top_n)

    return top_keywords_df, top_skills_df