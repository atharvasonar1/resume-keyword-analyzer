# Creates charts and word clouds for the results.
# Creates charts and word clouds for the project results.

import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for script use

OUTPUTS_DIR = "outputs"


def _ensure_outputs_dir():
    """Create the outputs folder if it does not exist."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def plot_keyword_frequency(keywords_df, job_title: str, top_n: int = 20):
    """
    Save a horizontal bar chart of the top frequency keywords.

    Parameters
    ----------
    keywords_df : DataFrame with columns ['keyword', 'count'] from keyword_frequency.py.
    job_title   : Used in the chart title and output filename.
    top_n       : How many keywords to show. Defaults to 20.

    Returns
    -------
    Path to the saved chart file.
    """
    _ensure_outputs_dir()

    if keywords_df.empty:
        print("Warning: keyword frequency DataFrame is empty. Skipping chart.")
        return None

    df = keywords_df.head(top_n).copy()
    df = df.sort_values("count", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["keyword"], df["count"], color="steelblue")
    ax.set_xlabel("Count")
    ax.set_title(f"Top {top_n} Keywords — {job_title}")
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()

    safe_title = job_title.lower().replace(" ", "_")
    filename = os.path.join(OUTPUTS_DIR, f"{safe_title}_keyword_frequency.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved keyword frequency chart: {filename}")
    return filename


def plot_skill_frequency(skills_df, job_title: str, top_n: int = 20):
    """
    Save a horizontal bar chart of the top skill keywords.

    Parameters
    ----------
    skills_df : DataFrame with columns ['skill', 'count'] from keyword_frequency.py.
    job_title : Used in the chart title and output filename.
    top_n     : How many skills to show. Defaults to 20.

    Returns
    -------
    Path to the saved chart file.
    """
    _ensure_outputs_dir()

    if skills_df.empty:
        print("Warning: skill frequency DataFrame is empty. Skipping chart.")
        return None

    df = skills_df.head(top_n).copy()
    df = df.sort_values("count", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["skill"], df["count"], color="darkorange")
    ax.set_xlabel("Count")
    ax.set_title(f"Top {top_n} Skills — {job_title}")
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()

    safe_title = job_title.lower().replace(" ", "_")
    filename = os.path.join(OUTPUTS_DIR, f"{safe_title}_skill_frequency.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved skill frequency chart: {filename}")
    return filename


def plot_tfidf_terms(tfidf_df, job_title: str, top_n: int = 20):
    """
    Save a horizontal bar chart of the top TF-IDF terms.

    Parameters
    ----------
    tfidf_df  : DataFrame with columns ['term', 'mean_tfidf'] from tfidf_analysis.py.
    job_title : Used in the chart title and output filename.
    top_n     : How many terms to show. Defaults to 20.

    Returns
    -------
    Path to the saved chart file.
    """
    _ensure_outputs_dir()

    if tfidf_df.empty:
        print("Warning: TF-IDF DataFrame is empty. Skipping chart.")
        return None

    df = tfidf_df.head(top_n).copy()
    df = df.sort_values("mean_tfidf", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["term"], df["mean_tfidf"], color="seagreen")
    ax.set_xlabel("Mean TF-IDF Score")
    ax.set_title(f"Top {top_n} TF-IDF Keywords — {job_title}")
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()

    safe_title = job_title.lower().replace(" ", "_")
    filename = os.path.join(OUTPUTS_DIR, f"{safe_title}_tfidf_keywords.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved TF-IDF chart: {filename}")
    return filename


def plot_wordcloud(keywords_df, job_title: str, word_col: str = "keyword", weight_col: str = "count"):
    """
    Save a word cloud image built from keyword counts.

    Falls back gracefully if the wordcloud package is not installed.

    Parameters
    ----------
    keywords_df : DataFrame with a word column and a numeric weight column.
    job_title   : Used in the chart title and output filename.
    word_col    : Column name for the words. Defaults to 'keyword'.
    weight_col  : Column name for the weights. Defaults to 'count'.

    Returns
    -------
    Path to the saved image, or None if wordcloud is unavailable or input is empty.
    """
    _ensure_outputs_dir()

    if keywords_df.empty:
        print("Warning: keyword DataFrame is empty. Skipping word cloud.")
        return None

    try:
        from wordcloud import WordCloud
    except ImportError:
        print("wordcloud package not installed. Skipping word cloud.")
        return None

    word_freq = dict(zip(keywords_df[word_col], keywords_df[weight_col]))

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="Blues",
        max_words=100,
    ).generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Keyword Word Cloud — {job_title}", fontsize=14)
    plt.tight_layout()

    safe_title = job_title.lower().replace(" ", "_")
    filename = os.path.join(OUTPUTS_DIR, f"{safe_title}_wordcloud.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved word cloud: {filename}")
    return filename


def save_all_charts(keywords_df, skills_df, tfidf_df, job_title: str, top_n: int = 20):
    """
    Convenience function: generate and save all charts in one call.

    Parameters
    ----------
    keywords_df : Top keywords DataFrame from keyword_frequency.py.
    skills_df   : Top skills DataFrame from keyword_frequency.py.
    tfidf_df    : Top TF-IDF terms DataFrame from tfidf_analysis.py.
    job_title   : Role name used in titles and filenames.
    top_n       : Number of items per chart.

    Returns
    -------
    List of file paths that were saved successfully.
    """
    saved = []

    path = plot_keyword_frequency(keywords_df, job_title, top_n=top_n)
    if path:
        saved.append(path)

    path = plot_skill_frequency(skills_df, job_title, top_n=top_n)
    if path:
        saved.append(path)

    path = plot_tfidf_terms(tfidf_df, job_title, top_n=top_n)
    if path:
        saved.append(path)

    path = plot_wordcloud(keywords_df, job_title)
    if path:
        saved.append(path)

    print(f"\n{len(saved)} chart(s) saved to '{OUTPUTS_DIR}/'.")
    return saved