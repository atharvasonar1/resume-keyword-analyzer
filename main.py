from src.data_loader import filter_by_job_title, inspect_dataset, load_jobs_csv


def main():
    print("Resume Keyword Analyzer")
    print("Loading job postings dataset...")
    print()

    try:
        jobs_df = load_jobs_csv()
        inspect_dataset(jobs_df)

        print()
        search_term = input("Enter a job title to search for: ").strip()

        if search_term == "":
            print("No job title entered. Please run the program again with a job title.")
            return

        matching_jobs = filter_by_job_title(jobs_df, search_term)
        match_count = len(matching_jobs)

        print()
        print(f"Matching postings for '{search_term}': {match_count}")

        if match_count == 0:
            print("No matching job postings were found.")
        else:
            print()
            print("Sample matching job titles:")
            for title in matching_jobs["title"].head(5):
                print(f"- {title}")
    except FileNotFoundError as error:
        print(error)
    except ValueError as error:
        print(error)


if __name__ == "__main__":
    main()
