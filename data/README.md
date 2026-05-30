# Data Folder

Place the job postings CSV file in this folder manually.

The expected filename for the project is:

```text
data/jobs.csv
```

A preferred dataset is the Kaggle LinkedIn Job Postings 2023-2024 dataset.

Expected fields include:

- Job title
- Company
- Location
- Job description

The exact column names may vary depending on the dataset. For example, the job title column might be named `title`, `job_title`, or something similar. These column names will be confirmed when dataset loading is implemented.

Large datasets should not be committed to GitHub. The `.gitignore` file excludes CSV files in this folder.
