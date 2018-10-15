import cv2
import os
import pandas as pd
import numpy as np

cdir = os.getcwd()
path = "label/img"
train_path = ["data/train/"+str(x) for x in range(10)]
val_path = ["data/val/"+str(x) for x in range(10)]
df = pd.read_csv('label.csv', header=None, dtype=object)

for dir in train_path+val_path:
    if not os.path.exists(dir):
        os.makedirs(dir)

for i in range(1000):
    out_dir = [val_path, train_path]
    sel = np.random.binomial(1, 0.8)
    img = cv2.imread(path + '/'+str(i+1)+'.jpg')
    tmp = df.values[i]
    for j, w in enumerate(range(0, 138, 23)):
        crop_img = img[:, w:w+23]
        label = tmp[0][j]
        cv2.imwrite(os.path.join(out_dir[sel][int(label)], str(i)+'_'+label+'.jpg'), crop_img)