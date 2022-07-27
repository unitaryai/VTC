import os
import pickle
import sys
import time

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

_filedir = os.path.dirname(os.path.realpath(__file__))
_base = os.path.realpath(os.path.join(_filedir, ".."))
sys.path.insert(0, _base)

CSV = ""
ROOT = ""
CSV = os.path.join(_base, CSV)
BATCH_SIZE = 24 * 4
DEVICE = "cuda:2"

model, preproc = clip.load("ViT-B/32", device="cpu", jit=False)
model.to(DEVICE)
model = model.float()

assert model.training is False

df = pd.read_csv(CSV)

files = [x[len("results/") : -4] + ".jpg" for x in df.video_path]
filenames = [os.path.join(ROOT, x) for x in files]


class DS(Dataset):
    def __init__(self, fns):
        self.filenames = fns

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        jpg = self.filenames[index]
        img = Image.open(jpg)
        return preproc(img)


ds = DS(filenames)

dl = DataLoader(
    ds, batch_size=BATCH_SIZE, num_workers=13, shuffle=False, pin_memory=True
)


out = []

tic = time.time()
for batchi, batch in enumerate(dl):
    imgs = batch
    imgs = imgs.to(DEVICE)

    with torch.no_grad():
        y = model.encode_image(imgs)

    out.append(y.cpu().numpy())

    toc = time.time() - tic
    tic = time.time()

    print(batchi, "/", len(dl), "%.1fHz" % (BATCH_SIZE / toc), y.shape)

stacked = np.vstack(out)
stacked = torch.tensor(stacked)
reddit_ids = torch.tensor(df.reddit_id.tolist())

save_dict = {"reddit_ids": reddit_ids, "embeddings": stacked}

torch.save(save_dict, "clip_vit_embeddings.pth")
