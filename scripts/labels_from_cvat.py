# from warnings import warn
from logging import warn
import os
import shutil
import numpy as np
from tqdm import tqdm
from scripts.fixball_labels import fixlabel
import argparse
from glob import glob
from os.path import join, basename, splitext
from shutil import copyfile
from collections import defaultdict


# labels: (not used) ["ball", "robot", "cross", "goalpost", "center"]
# labelperm = [0, 2, 1, None, 3]  # for training with cener
# labelperm = [0, 2, 1, None, None]  # for training
# labelperm = [0, None, None, None, None]  # for training single class
labelperm = [0, None, 1, None, None]  # for ball-cross
# labelperm = [0, None, None, None, None]  # for test circle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="../spldata/")
    parser.add_argument("--camera", type=str, default="bottom", choices=["top", "bottom"])
    parser.add_argument("--min-obj-size", type=float, default=0.0)
    parser.add_argument("--filter-void", default=False, action="store_true")
    parser.add_argument("--random_accept", default=0.1)
    args = parser.parse_args()
    dirs = [d for d in os.listdir(args.dir) if not d.startswith("cvat")]


    if args.camera == "bottom":
        args.min_obj_size = 0.15
        args.random_accept = 0.0
        args.filter_void = True

    labs = glob(join(args.dir, "cvat/**/*.txt"), recursive=True)
    images = glob(join(args.dir, "datasets/**/*"), recursive=True)
    file2lab = dict()
    file2im = defaultdict(list)
    file = "hello"
    for f in labs:
        key = basename(f).replace(".txt", "")
        file2lab[key] = f

    for f in images:
        for ext in (".jpg", ".png"):
            if f.endswith(ext):
                key = basename(f).replace(ext, "")
                file2im[key].append(f)    

    for key, impaths in tqdm(file2im.items()):
        for impath in impaths:
            # create directory of dst if it doesn't exist
            dstdir = os.path.dirname(impath).replace("\\", "/")

            if args.camera == "bottom":
                dataset = dstdir.split("/")[-3]
                dstdir = dstdir.replace(dataset, dataset + "_bottom")

            os.makedirs(dstdir, exist_ok=True)
            os.makedirs(dstdir.replace("/images/", "/labels/"), exist_ok=True)
                
            if key not in file2lab:
                warn(f"{key} not in file2lab")
                continue
            else:
                lab = file2lab[key]
            boxes = []
            for line in open(lab, "r"):
                if line == "\n":
                    continue
                obj, *box = line.split()
                obj = labelperm[int(obj)]
                if obj is None:
                    continue
                if obj == labelperm[0]:
                    box, cond = fixlabel([float(b) for b in box])
                elif obj == labelperm[1]:  # make 2.5x larger to help detector
                    scale = 1.0
                    x, y, w, h = [float(b) for b in box]
                    box = [x, y, scale * w, scale * h]
                _, _, w, h = [float(x) for x in box]
                area = max(w, h)
                if ("bottom" not in key) and area < args.min_obj_size and (np.random.rand() > args.random_accept):
                    continue
                line = " ".join([str(obj)] + [str(b) for b in box])
                boxes.append(line.replace("\n", ""))
                
            if len(boxes) > 0 or not args.filter_void:
                with open(dstdir.replace("/images/", "/labels/") + "/" + key + ".txt", "w") as io:
                    io.write("\n".join(boxes))
                if args.camera == "bottom":
                    shutil.copy(impath, dstdir)