from src.data_loader import inspect_dataset, load_jobs_csv


def main():
    print("Resume Keyword Analyzer")
    print("Loading job postings dataset...")
    print()

    try:
        jobs_df = load_jobs_csv()
        inspect_dataset(jobs_df)
    except FileNotFoundError as error:
        print(error)


if __name__ == "__main__":
    main()
