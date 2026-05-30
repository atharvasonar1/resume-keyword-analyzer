# Resume Keyword Analyzer

## Objective

Resume Keyword Analyzer is a CSCI 185 Data Mining final project that will analyze job posting data and identify important keywords and skills for a selected job title.

The project is designed to help users compare a target role, such as data analyst, software engineer, or data scientist, against real job postings to understand which skills appear most often.

## Simple User Flow

1. The user places a job postings CSV file in the `data/` folder.
2. The user selects or enters a job title to analyze.
3. The project filters job postings related to that role.
4. The project will generate keyword frequency results, TF-IDF keywords, skill association rules, and simple charts.
5. Results and charts will be saved in the `outputs/` folder.

## Dataset Description

This project will use a CSV dataset of job postings. A preferred dataset is the Kaggle LinkedIn Job Postings 2023-2024 dataset.

The dataset should be downloaded manually and placed at:

```text
data/jobs.csv
```

Expected fields include:

- Job title
- Company
- Location
- Job description

The exact column names may vary depending on the dataset and will be confirmed during the dataset loading step.

Large datasets should be downloaded manually and should not be committed to GitHub. CSV files in the `data/` folder are ignored by `.gitignore`.

## Planned Features

- Load job postings from a CSV file.
- Filter postings by selected job title.
- Clean and prepare job descriptions for text analysis.
- Calculate top frequent keywords and skills.
- Identify role-specific keywords using TF-IDF.
- Find common skill combinations using association rule mining.
- Create simple charts and word clouds for the results.

## Repository Structure

```text
resume-keyword-analyzer/
├── data/
│   └── README.md
├── notebooks/
│   └── README.md
├── outputs/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── keyword_frequency.py
│   ├── tfidf_analysis.py
│   ├── association_rules.py
│   └── visualizations.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

- `data/`: will store the job postings CSV file locally.
- `data/README.md`: explains where to place the dataset.
- `notebooks/`: will contain the main analysis notebook for the project workflow.
- `notebooks/README.md`: describes the purpose of the notebook folder.
- `outputs/`: will store generated charts, word clouds, and result files.
- `outputs/README.md`: describes the expected output files.
- `src/__init__.py`: marks `src` as a Python package.
- `src/data_loader.py`: will handle loading the job postings CSV file.
- `src/preprocessing.py`: will clean and prepare job descriptions for text analysis.
- `src/keyword_frequency.py`: will calculate the most common words and skills for a selected role.
- `src/tfidf_analysis.py`: will identify important role-specific keywords using TF-IDF.
- `src/association_rules.py`: will find common skill combinations using association rule mining.
- `src/visualizations.py`: will create charts and word clouds for the results.
- `main.py`: will be the simple command-line entry point for the project.
- `requirements.txt`: lists the Python packages needed for the project.
- `.gitignore`: excludes local files, datasets, caches, and generated outputs from Git.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Future Run Instructions

After the analysis code is implemented, the project will be run from the command line:

```bash
python main.py
```

The user will provide a job title, and the project will analyze matching job postings from the CSV file in the `data/` folder.

## Team

- Atharva Sonar
- Sricharan2005
