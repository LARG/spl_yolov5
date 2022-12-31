import argparse
import os

# 0 -> ball=0
# 1 -> cross=2
# others to none

# mapping to cvat project
labelperm = [0, 2, None, None, None] 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=str)
    parser.add_argument("--o", type=str)
    args = parser.parse_args()

    os.makedirs(args.o, exist_ok=True)
    files = os.listdir(args.i)
    fname = os.path.basename(args.i)
    for f in files:
        with open(os.path.join(args.i, f), "r") as io:
            labs = io.readlines()
        new_labs = ''
        for l in labs:
            l = l.split()
            subs = labelperm[int(l[0])]
            if subs is not None:
                newl = str(subs) + " " + " ".join(l[1:])
                new_labs += newl.replace("\n", "").strip() + "\n"
        with open(os.path.join(args.o, f).replace("\n", "").strip(), "w") as io:
            io.write(new_labs)
