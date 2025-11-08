# ingestion/load_dataset.py
import os
from pathlib import Path

DATA_PATH_LOCAL = "data/Finance_FAQ_Extended.json"

def verify_dataset_exists(path=DATA_PATH_LOCAL):
    """Checks if the dataset file exists at the specified path."""
    if os.path.exists(path):
        print(f"Dataset found at {path}.")
    else:
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            f"Please ensure the file 'Finance_FAQ_Extended.json' is in the 'data' directory."
        )

if __name__ == "__main__":
    verify_dataset_exists()
