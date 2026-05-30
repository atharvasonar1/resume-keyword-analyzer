from pathlib import Path

import pandas as pd


def load_jobs_csv(file_path="data/jobs.csv"):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}\n"
            "Place the job postings CSV file at data/jobs.csv before running the project."
        )

    df = pd.read_csv(path)

    title_column, description_column = find_likely_columns(df)
    for column in [title_column, description_column]:
        if column is not None:
            df[column] = df[column].fillna("")

    return df


def find_likely_columns(df):
    title_options = ["title", "job_title", "job title", "jobtitle"]
    description_options = [
        "description",
        "job_description",
        "job description",
        "jobdescription",
        "description_text",
    ]

    title_column = None
    description_column = None

    for column in df.columns:
        clean_column = column.strip().lower()

        if title_column is None and clean_column in title_options:
            title_column = column

        if description_column is None and clean_column in description_options:
            description_column = column

    return title_column, description_column


def filter_by_job_title(df, search_term, title_column="title"):
    if title_column not in df.columns:
        raise ValueError(f"Title column not found: {title_column}")

    matches = df[title_column].astype(str).str.contains(search_term, case=False, na=False)

    return df[matches].copy()


def inspect_dataset(df):
    rows, columns = df.shape
    title_column, description_column = find_likely_columns(df)

    print(f"Rows: {rows}")
    print(f"Columns: {columns}")
    print()

    print("Column names:")
    for column in df.columns:
        print(f"- {column}")
    print()

    if title_column is not None:
        print(f"Likely title column: {title_column}")
        print(f"Missing title values: {df[title_column].isna().sum()}")
    else:
        print("Likely title column: not found")

    if description_column is not None:
        print(f"Likely description column: {description_column}")
        print(f"Missing description values: {df[description_column].isna().sum()}")
    else:
        print("Likely description column: not found")
