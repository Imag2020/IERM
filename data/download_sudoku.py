import os
from huggingface_hub import hf_hub_download

DATA_DIR = "data/sudoku"

FILES = [
    "train.npz",
    "test.npz"
]

REPO_ID = "sapientinc/sudoku-extreme"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for file in FILES:
        print(f"Downloading {file}...")

        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=file,
            repo_type="dataset"
        )

        target = os.path.join(DATA_DIR, file)

        if not os.path.exists(target):
            os.rename(path, target)

    print("\nSudoku dataset ready.")
    print("Location:", DATA_DIR)


if __name__ == "__main__":
    main()