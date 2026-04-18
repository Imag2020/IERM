import os
import csv
from typing import Dict, Optional

import numpy as np
from huggingface_hub import hf_hub_download


def download_sudoku_data(
    output_dir: str = "data/sudoku",
    source_repo: str = "sapientinc/sudoku-extreme",
    min_difficulty: Optional[int] = None,
) -> Dict[str, str]:
    """
    Download Sudoku CSV files from Hugging Face and convert them locally to:
      - data/sudoku/train.npz
      - data/sudoku/test.npz

    Returns:
        dict with keys {"train", "test"} pointing to the generated NPZ files.
    """
    os.makedirs(output_dir, exist_ok=True)

    paths: Dict[str, str] = {}

    for split in ["train", "test"]:
        npz_path = os.path.join(output_dir, f"{split}.npz")

        if os.path.exists(npz_path):
            print(f"[Sudoku] Reusing existing file: {npz_path}")
            paths[split] = npz_path
            continue

        print(f"[Sudoku] Downloading {split}.csv from {source_repo} ...")
        csv_path = hf_hub_download(
            repo_id=source_repo,
            filename=f"{split}.csv",
            repo_type="dataset",
        )

        inputs = []
        labels = []

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)

            for row in reader:
                # Expected columns from your notebook:
                # source, q, a, rating
                if len(row) != 4:
                    raise ValueError(
                        f"Unexpected row format in {split}.csv: expected 4 columns, got {len(row)}"
                    )

                source, q, a, rating = row

                if min_difficulty is not None and int(rating) < min_difficulty:
                    continue

                if len(q) != 81 or len(a) != 81:
                    raise ValueError(
                        f"Unexpected puzzle length in {split}.csv: len(q)={len(q)}, len(a)={len(a)}"
                    )

                inp = np.frombuffer(
                    q.replace(".", "0").encode("ascii"),
                    dtype=np.uint8,
                ).reshape(9, 9) - ord("0")

                lab = np.frombuffer(
                    a.encode("ascii"),
                    dtype=np.uint8,
                ).reshape(9, 9) - ord("0")

                inputs.append(inp)
                labels.append(lab)

        if not inputs:
            raise RuntimeError(
                f"No Sudoku samples were loaded for split='{split}'. "
                f"Check min_difficulty or dataset format."
            )

        inputs_np = np.stack(inputs).astype(np.uint8)
        labels_np = np.stack(labels).astype(np.uint8)

        np.savez_compressed(npz_path, inputs=inputs_np, labels=labels_np)
        print(f"[Sudoku] {split}: saved {len(inputs_np)} puzzles -> {npz_path}")

        paths[split] = npz_path

    print("[Sudoku] Dataset ready.")
    return paths


if __name__ == "__main__":
    download_sudoku_data()