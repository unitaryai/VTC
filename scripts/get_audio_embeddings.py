import os
import pickle
import sys
import time
from pathlib import Path

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(os.path.realpath(__file__)).parents[1]) + "/")

import av
from GDT.datasets.audio_utils import load_audio
from GDT.datasets.decoder import get_start_end_idx
from GDT.model import AudioBaseNetwork, Identity
from model.model import MLP

_filedir = os.path.dirname(os.path.realpath(__file__))
_base = os.path.realpath(os.path.join(_filedir, ".."))
sys.path.insert(0, _base)

ROOT = ""
CSV = ""
CSV = os.path.join(_base, CSV)
BATCH_SIZE = 24 * 4
DEVICE = "cuda:0"
NUM_CLIPS = 5

audio_model = AudioBaseNetwork("resnet9", pretrained=True, duration=1)
weights = torch.load("gdt_IG65M.pth", map_location="cpu")
audio_weights = {}
for k, v in weights["model"].items():
    if "audio" in k:
        audio_weights[k.split("audio_network.")[1]] = v
audio_weights["base.fc.weight"] = audio_model.base.fc.weight
audio_weights["base.fc.bias"] = audio_model.base.fc.bias
audio_model.load_state_dict(audio_weights)
audio_model.base.fc = Identity()

audio_model.to(DEVICE)
model = audio_model.float()

df = pd.read_csv(CSV)

files = [x[len("results/") : -4] + ".mp4" for x in df.video_path]
filenames = [os.path.join(ROOT, x) for x in files]


class DS(Dataset):
    def __init__(self, fns):
        self.filenames = fns

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        audio_present = True
        video = self.filenames[index]
        try:
            container = av.open(video)
        except:
            try:
                container = av.open(video, metadata_errors="ignore")
            except Exception as e:
                print(f"vid open failed with: {e}")
                container = None
                audio_present = False

        # if multi_thread_decode:
        # Enable multiple threads for decoding.
        if container is not None:
            try:
                container.streams.video[0].thread_type = "AUTO"
                fps = float(container.streams.video[0].average_rate)
                frames_length = container.streams.video[0].frames
                duration = container.streams.video[0].duration
            except:
                if len(container.streams.audio) > 0:
                    container.streams.audio[0].thread_type = "AUTO"
                    if container.streams.audio[0].average_rate is not None:
                        fps = float(container.streams.audio[0].average_rate)
                    else:
                        fps = 30
                    frames_length = container.streams.audio[0].frames
                    duration = container.streams.audio[0].duration
                else:
                    print(f"No container.streams.audio for {video}")
                    audio_present = False
            if audio_present:
                audio_clips = []
                time_points = [0.15, 0.3, 0.45, 0.6, 0.85]
                # time_points = [0.25, 0.5, 0.75]
                # time_points = [0.5]
                for time_point in time_points:
                    # fr_sec = (frames_length / 2) / fps
                    fr_sec = frames_length * time_point / fps
                    audio = load_audio(
                        video,
                        fr_sec=fr_sec,
                        aug_audio=[],
                        num_sec=2,
                        sample_rate=24000,
                        aud_spec_type=2,
                        use_volume_jittering=False,
                        use_temporal_jittering=False,
                        z_normalize=False,
                    )
                    if audio is None:
                        audio = torch.ones((1, 257, 199))
                    audio_clips.append(audio)
                audio_clips = torch.cat(audio_clips, axis=0)
        if not audio_present or audio is None or audio.shape[2] != 199:
            audio_clips = torch.ones((NUM_CLIPS, 257, 199))

        return audio_clips


ds = DS(filenames)

dl = DataLoader(
    ds, batch_size=BATCH_SIZE, num_workers=13, shuffle=False, pin_memory=True
)


out = []

tic = time.time()
for batchi, batch in tqdm(enumerate(dl), total=len(dl)):
    audio = batch
    audio = audio.to(DEVICE)

    with torch.no_grad():
        if audio.shape[1] != 1:
            audio = audio.permute(1, 0, 2, 3)
            y = torch.stack([model(a.unsqueeze(1)) for a in audio], axis=0)
            y = y.permute(1, 0, 2)
        else:
            y = model(audio)

    out.append(y.cpu().numpy())

    toc = time.time() - tic
    tic = time.time()

    print(batchi, "/", len(dl), "%.1fHz" % (BATCH_SIZE / toc), y.shape)

stacked = np.vstack(out)
stacked = torch.tensor(stacked)
reddit_ids = torch.tensor(df.reddit_id.tolist())

save_dict = {"reddit_ids": reddit_ids, "embeddings": stacked}

torch.save(save_dict, "audio_embeddings_no_aug_5clip_5embeds_2sec.pth")
