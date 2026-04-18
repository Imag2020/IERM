import os
import subprocess
import shutil

DATA_DIR = "data/arc_agi1"
TMP_DIR = "/tmp/arc_agi"


def main():

    if os.path.exists(DATA_DIR):
        print("ARC dataset already exists.")
        return

    print("Cloning ARC repository...")

    subprocess.run(
        ["git", "clone", "https://github.com/fchollet/ARC-AGI", TMP_DIR],
        check=True
    )

    os.makedirs(DATA_DIR, exist_ok=True)

    shutil.copytree(
        os.path.join(TMP_DIR, "data", "training"),
        os.path.join(DATA_DIR, "training")
    )

    shutil.copytree(
        os.path.join(TMP_DIR, "data", "evaluation"),
        os.path.join(DATA_DIR, "evaluation")
    )

    print("ARC dataset ready.")
    print("Location:", DATA_DIR)


if __name__ == "__main__":
    main()