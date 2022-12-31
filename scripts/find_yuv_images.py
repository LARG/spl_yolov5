import argparse
import os
import glob
import shutil

# this is easier to do with bash

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=str)
    parser.add_argument("--o", type=str)
    args = parser.parse_args()
    os.makedirs(args.o, exist_ok=True)
    files = glob.glob(f"{args.i}/**/*.yuv", recursive=True)
    for f in files:
        ftgt = f.replace(args.i, "").replace("\\", "/").replace("/", "_")
        shutil.copy(f, os.path.join(args.o, ftgt))
