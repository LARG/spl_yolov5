# %%
# this script fixes the labels where the ball is not complete
# it might fail when the ball is in a corner, so one prerequisite
# is that the ball is that there's a gap either horizontally or vertically
# the output are new bounding boxes that may exceed the image size

from typing import List, Tuple, Optional
from glob import glob
from os.path import join, basename
from shutil import copy
from os import makedirs
from tqdm import tqdm
from collections import Counter
import datetime
import yaml
import numpy as np


# %%
def approx(x: float, y: float, tol: float = 0.01):
    return np.abs(x - y) < tol


def fixlabel(box: List[float], ar=.75) -> Tuple[List[float], str]:
    x, y, w, h = box
    left, right = x - 0.5 * w, x + 0.5 * w
    up, down = y - 0.5 * h, y + 0.5 * h

    if (
        not approx(left, 0.)
        and not approx(right, 1.)
        and not approx(up, 0.)
        and not approx(down, 1.)
    ):
        # no fix necessary
        return box, "within"
    elif (
        (approx(left, 0.) or approx(right, 1.))
        and (approx(up, 0.) or approx(down, 1.))
    ):
        # corner (invalid) case
        return box, "corner"
    elif approx(left, 0.):
        new_w = max(h * ar, w)
        new_x = x - 0.5 * (new_w - w)
        return [new_x, y, new_w, h], "left"
    elif approx(right, 1.0):
        new_w = max(h * ar, w)
        new_x = x + 0.5 * (new_w - w)
        return [new_x, y, new_w, h], "right"
    elif approx(up, 0.):
        new_h = max(w / ar, h)
        new_y = y - 0.5 * (new_h - h)
        return [x, new_y, w, new_h], "upper" 
    elif approx(down, 1.0):
        new_h = max(h, w / ar)
        new_y = y + 0.5 * (new_h - h)
        return [x, new_y, w, new_h], "bottom"
    else:
        raise Exception("shouldn't be here")


if __name__ == "__main__":

    # %%
    # tgtdir = "../../../spl_logs/labels"
    # tgtdir = "../../../spl_synthetic/labels"
    tgtdir = "../../../naodevils/labels"
    # tgtdir = "../../../naodevils_auto/labels"
    tgtobj = "0"
    labels = glob(join(tgtdir, "**/*.txt"))
    N = len(labels)
    print(f"Found {N} labels.")
        

    # %%
    fixes = []
    write = True

    for lab in tqdm(labels):
        boxes = []
        has_fixes = False
        for i, line in enumerate(open(lab, "r")):
            obj, *box = line.split()
            if obj == tgtobj:
                prev_box = [float(b) for b in box]
                new_box, cond = fixlabel(prev_box)
                if cond != "within":
                    row = dict(
                        index=i,
                        prev_box=prev_box,
                        new_box=new_box,
                        cond=cond,
                        files=lab
                    )
                    fixes.append(row)
                    has_fixes = True
                    box = [str(b) for b in new_box]
            boxes.append(" ".join([obj] + box))
        if write and has_fixes:
            with open(lab, "w") as io:
                io.write("\n".join(boxes))


    #%%
    condition = Counter([x["cond"] for x in fixes])
    print(condition)

    #%%
    dt = datetime.datetime.now()
    datestr = f"{dt.year}{dt.month:02d}{dt.day:02d}"
    timestr = f"{dt.hour:02d}{dt.minute:02d}{dt.second:02d}"
    logfile = f"fixlabels_{datestr}_{timestr}_log.yaml"
    with open(logfile, "w") as io:
        yaml.dump(fixes, io)

    # %%
