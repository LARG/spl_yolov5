import argparse
import os
import glob
import numpy as np
# import yuvio
from PIL import Image
from tqdm import tqdm
import cv2


wtop, htop = 1280, 960
# wtop, htop = 960, 1280
wbot, hbot = 320, 240
# wbot, hbot = 240, 320
BYTES_TO_IGNORE = 36


b_pre = np.array([-16, -128, -16, -128])
W = np.array(
    [
        [1.164,    0,   0,  1.596], 
        [1.164, -0.391,   0, -0.813],
        [1.164,  2.018,   0,  0],
        [  0,    0, 1.164,  1.596],
        [  0, -0.391, 1.164, -0.813],
        [  0,  2.018, 1.164,  0] 
    ]  
)

# W = np.array(
#     [
#         [1.164,    0,   0,  409], 
#         [1.164, -100,   0, -209],
#         [1.164,  516,   0,  128],
#         [  0,    0, 1.164,  409],
#         [  0, -100, 1.164, -209],
#         [  0,  516, 1.164,  128] 
#     ]  
# )
W = W.transpose() / 255.0**2
b = np.array([0, 128, 0, 0, 128, 0]) / 255.0**2


def clip255(x: int):
    return np.clip(x, 0, 255)


def yuv444ToRgb(y, u, v):
    C = y - 16
    D = u - 128
    E = v - 128
    r = clip255(( 298 * C           + 409 * E + 128) >> 8)
    g = clip255(( 298 * C - 100 * D - 208 * E + 128) >> 8)
    b = clip255(( 298 * C + 516 * D           + 128) >> 8)
    return [r, g, b]


def yuyv2rgb(x: np.ndarray):
    h, w = x.shape[:2]
    x = (x.reshape(-1, 4) + b_pre) @ W + b
    x = np.clip(x, 0.0, 1.0)
    x = x.reshape(h, 2 * w, 3)
    return (x * 255).astype(np.uint8)


def rgb2yuyv(x: np.ndarray):
    img_yuv = cv2.cvtColor(x, cv2.COLOR_RGB2YUV)
    y0 = np.expand_dims(img_yuv[...,0][::,::2], axis=2)
    u = np.expand_dims(img_yuv[...,1][::,::2], axis=2)
    y1 = np.expand_dims(img_yuv[...,0][::,1::2], axis=2)
    v = np.expand_dims(img_yuv[...,2][::,::2], axis=2)
    img_yuyv = np.concatenate((y0, u, y1, v), axis=2)
    return img_yuyv



def bgr2yuyv(x: np.ndarray):
    return rgb2yuyv(x[..., ::-1])


 
def main(args: argparse.Namespace):
    os.makedirs(args.o, exist_ok=True)
    file = glob.glob(f"{args.i}/**/*.yuv", recursive=True)
    for f in tqdm(file):
        if args.what == "bottom" and "top_" in f:
            continue
        elif args.what == "top" and "bottom_" in f:
            continue
        w, h = (wtop, htop) if ("top_" in f ) else (wbot, hbot)
        yuyv = []
        # im = yuvio.imread(f, w, h, "yuyv422")
        # # im = yuvio.imread(f, w, h, "yuv444p")
        # X = np.stack([im.y[:, ::2], im.u, im.y[:, 1::2], im.v], -1)
        # rgb = []
        with open(f, "rb") as fp:
            X = fp.read(BYTES_TO_IGNORE + (w // 2) * h * 4)
        X = np.array(list(X))[BYTES_TO_IGNORE:]
            # for i in range(0, len(bytes), 4):
            #     if i < BYTES_TO_IGNORE:
            #         continue
            #     yuyv.append(bytes[i])
            #     u.append(bytes[i + 1])
            #     y1.append(bytes[i + 2])
            #     v.append(bytes[i + 3])
                # rgb0 = yuv444ToRgb(y0[-1], u[-1], v[-1])
                # rgb1 = yuv444ToRgb(y1[-1], u[-1], v[-1])
                # rgb.append(rgb0)
                # rgb.append(rgb1)
        # rgb = np.array(rgb).reshape(h, w, 3)
        # X = np.stack((y0, u, y1, v), axis=-1)
        # X = X.reshape(h, w // 2, 4)
        X = (X.reshape(-1, 4) + b_pre) @ W + b
        X = np.clip(X, 0.0, 1.0)
        X = X.reshape(h, w, 3)
        imout = Image.fromarray((X * 255).astype(np.uint8))
        base = os.path.basename(f).replace(".yuv", ".jpg")
        basedir = f.replace("\\", "/").split("/")[-2]
        imout.save(os.path.join(args.o, basedir + "_" + base))
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=str)
    parser.add_argument("--o", type=str)
    parser.add_argument(
        "--what", type=str, default="both", choices=("both", "top", "bottom")
    )
    args = parser.parse_args()
    main(args)