# from warnings import warn
from logging import warn
import os
import shutil
import cv2
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
    parser.add_argument("--camera", type=str, default="top", choices=["top", "bottom"])
    parser.add_argument("--min-obj-size", type=float, default=0.0)
    parser.add_argument("--min-ball-size", type=float, default=0.1)
    parser.add_argument("--ball-rescale-factor", type=float, default=1)
    parser.add_argument("--filter-void", default=False, action="store_true")
    parser.add_argument("--random_accept", default=0.1)
    args = parser.parse_args()
    dirs = [d for d in os.listdir(args.dir) if not d.startswith("cvat")]

    # if args.camera == "bottom":
    #     args.min_obj_size = 0.1
        # args.random_accept = 0.0
        # args.filter_void = True

    labs = glob(join(args.dir, "cvat/**/*.txt"), recursive=True)
    images = []
    for x in glob(join(args.dir, "datasets/**/*"), recursive=True):
        bdir = os.path.dirname(x.replace("\\", "/"))
        is_img = any(x.lower().endswith(z) for z in ("jpg", "jpeg", "png"))
        if is_img and "_bottom" not in bdir and "_top" not in bdir:
            images.append(x)
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
            sizes = []
            for line in open(lab, "r"):
                if line == "\n":
                    continue
                obj, *box = line.split()
                obj = labelperm[int(obj)]
                if obj is None:
                    continue
                if args.camera != "bottom" and obj == labelperm[0]:
                    box, cond = fixlabel([float(b) for b in box])
                elif obj == labelperm[1]:  # make 2.5x larger to help detector
                    scale = 1.0
                    box = [x, y, scale * w, scale * h]
                x, y, w, h = [float(x) for x in box]
                sizes.append((obj, (x, y, w, h)))
                # if ("bottom" not in key) and area < args.min_obj_size and (np.random.rand() > args.random_accept): 
                #     continue
                line = " ".join([str(obj)] + [str(b) for b in box])
                boxes.append(line.replace("\n", ""))
                
            if len(boxes) > 0 or not args.filter_void:
                if args.camera != "bottom":
                    with open(dstdir.replace("/images/", "/labels/") + "/" + key + ".txt", "w") as io:
                        io.write("\n".join(boxes))
                    continue

                has_ball = any(obj == labelperm[0] for (obj, _) in sizes)
                if has_ball:
                    F = 1.0
                    ibox = None
                    for j, (obj, box) in enumerate(sizes):
                        size = max(*box[2:])
                        if size == 0:
                            continue
                        U = 1 + np.random.rand() * (args.ball_rescale_factor - 1)
                        F = max(F, U * args.min_ball_size / size)
                        if obj == labelperm[0]:
                            ibox = j
                else:
                    # keep as control
                    if args.camera == "bottom":
                        shutil.copy(impath, dstdir)
                    with open(dstdir.replace("/images/", "/labels/") + "/" + key + ".txt", "w") as io:
                        io.write("\n".join(boxes))
                    continue

                # hard part focused zoom and crop
                # find voz
                obj, (x, y, w, h) = sizes[ibox]
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                size = max(w, h)
                # zoom in this image and save!
                im = cv2.imread(impath)
                U = 1 + np.random.rand() * (args.ball_rescale_factor - 1)
            F = U * args.min_ball_size / size
                H, W, D = im.shape
                x_ = int(W * x)
                y_ = int(H * y)
                left = int(x_ -  x_ / F)
                right = int(x_ + (W - x_) / F)
                up = int(y_ - y_ / F)
                down = int(y_ + (H - y_) / F)
                im2 = im[up:down, left:right]
                im2 = cv2.resize(im2, (W, H))
                (x, y, w, h), cond = fixlabel([float(b) for b in (x, y, w, h)])
                new_boxes = [f"0 {x} {y} {w * F} {h * F}\n"]
                for i in range(len(sizes)):
                    obj, (x1, y1, w1, h1) = sizes[i]
                    if obj == labelperm[0]: # it's the ball
                        continue
                    else:
                        dx = x1 - x
                        dy = y1 - y
                        x1_ = x + dx * F
                        y1_ = y + dy * F
                        if 0 < x1_ < 1 or 0 < y1_ < 1:
                            new_boxes.append(f"{obj} {x1_} {y1_} {w1 * F} {h1 * F}\n")
                        else:
                            pass
                fname = f"{dstdir}/{key}.png"
                cv2.imwrite(fname, im2)
                fname = dstdir.replace("/images/", "/labels/") + f"/{key}.txt"
                with open(fname, "w") as io:
                    io.write("\n".join(new_boxes))
