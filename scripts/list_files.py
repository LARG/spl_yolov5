import argparse
import os

# this is easier to do with bash

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=str)
    args = parser.parse_args()

    files = os.listdir(args.i)
    fname = os.path.basename(args.i)
    with open(os.path.join(os.path.dirname(args.i), fname + ".txt"), "w") as io:
        io.writelines(files)
