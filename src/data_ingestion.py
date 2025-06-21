import os
import requests
import zipfile
import io
import csv

# Constants
URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
EXTRACT_DIR = "liar_data"
TSV_FILES = ["train.tsv", "test.tsv", "valid.tsv"]
CSV_FILES = ["train.csv", "test.csv", "valid.csv"]

# Column headers from LIAR dataset
COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job_title", "state_info",
    "party_affiliation", "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context"
]

def download_and_extract():
    print("Downloading dataset...")
    response = requests.get(URL)
    if response.status_code != 200:
        raise Exception("Failed to download dataset.")

    print("Extracting files...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    print("Extraction complete.")

def convert_tsv_to_csv():
    print("\n Converting TSV to CSV...")
    for tsv, csv_ in zip(TSV_FILES, CSV_FILES):
        tsv_path = os.path.join(EXTRACT_DIR, tsv)
        csv_path = os.path.join(EXTRACT_DIR, csv_)

        if not os.path.exists(tsv_path):
            print(f" Skipped (not found): {tsv_path}")
            continue

        with open(tsv_path, "r", encoding="utf-8") as tsv_file, \
             open(csv_path, "w", encoding="utf-8", newline='') as csv_file:
            reader = csv.reader(tsv_file, delimiter="\t")
            writer = csv.writer(csv_file)
            writer.writerow(COLUMNS)  # Add header
            for row in reader:
                if len(row) == len(COLUMNS):
                    writer.writerow(row)
        print(f" Saved: {csv_path}")

def main():
    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR)

    download_and_extract()
    convert_tsv_to_csv()

if __name__ == "__main__":
    main()
