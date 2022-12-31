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
    parser.add_argument("--min-ball-size", type=float, default=0.2)
    parser.add_argument("--ball-rescale-factor", type=float, default=2.0)
    args = parser.parse_args()
    dirs = [d for d in os.listdir(args.dir) if not d.startswith("cvat")]


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
            dataset = dstdir.split("/")[-3]
            dstdir = dstdir.replace(dataset, dataset + "_bottomalt")

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

                elif obj == labelperm[1]:  # make 2.5x larger to help detector
                    scale = 1.0
                    box = [x, y, scale * w, scale * h]
                x, y, w, h = [float(x) for x in box]

                sizes.append((obj, (x, y, w, h)))
                line = " ".join([str(obj)] + [str(b) for b in box])
                boxes.append(line.replace("\n", ""))
                
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
                shutil.copy(impath, dstdir)
                with open(dstdir.replace("/images/", "/labels/") + "/" + key + ".txt", "w") as io:
                    io.write("\n".join(boxes))
                continue

            # hard part focused zoom and crop

            obj, (x, y, w, h) = sizes[ibox]
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            size = max(w, h)

            # zoom in this image and save!
            im = cv2.imread(impath)
            U = 1 + np.random.rand() * (args.ball_rescale_factor - 1)
            F = max(1, U * args.min_ball_size / size)
            H, W, D = im.shape
            
            Wf = int(W / F)
            Hf = int(H / F)
            r0 = int(H * y * (1.0 - 1.0 / F))
            c0 = int(W * x * (1.0 - 1.0 / F))

            im2 = im[r0:(r0 + Hf), c0:(c0 + Wf)]
            im2 = cv2.resize(im2, (W, H))
            (xfix, yfix, wfix, hfix), cond = fixlabel([float(b) for b in (x, y, w * F, h * F)])
            new_boxes = [f"0 {xfix} {yfix} {wfix} {hfix}\n"]
            # new_boxes = [f"0 {x} {y} {w * F} {h * F}\n"]
            for i in range(len(sizes)):
                if i == ibox:
                    continue
                obj, (x1, y1, w1, h1) = sizes[i]
                x1_ = (x1 * W - c0) / Wf
                y1_ = (y1 * H - r0) / Hf
                if 0 < x1_ < 1 and 0 < y1_ < 1:
                    new_boxes.append(f"{obj} {x1_} {y1_} {w1 * F} {h1 * F}\n")
                else:
                    pass
            fname = f"{dstdir}/{key}.png"
            cv2.imwrite(fname, im2)
            fname = dstdir.replace("/images/", "/labels/") + f"/{key}.txt"
            with open(fname, "w") as io:
                io.writelines(new_boxes)
