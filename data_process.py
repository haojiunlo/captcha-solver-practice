import cv2
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

cdir = os.getcwd()
path = "label/img"
train_path = ["data/train/" + str(x) for x in range(10)]
val_path = ["data/val/" + str(x) for x in range(10)]
df = pd.read_excel('label/label.xlsx', header=None, dtype=object)

for dir in train_path + val_path:
    if not os.path.exists(dir):
        os.makedirs(dir)

for file in tqdm(glob.glob("label/img/*.jpg")):
    out_dir = [val_path, train_path]
    sel = np.random.binomial(1, 0.8)
    img = cv2.imread(file)
    idx = int(file.split("/")[-1].split(".")[0])
    labels = df.iloc[idx - 1, 0]

    img_w = img.shape[1]

    # split 6 digits
    digit_w = img.shape[1] // 6
    for j, w in enumerate(range(0, img_w - 2, digit_w)):
        crop_img = img[:, w : w + digit_w]
        assert crop_img.shape[1] == digit_w
        label = labels[j]
        cv2.imwrite(os.path.join(out_dir[sel][int(label)], str(idx) + '_' + label + '.jpg'), crop_img)