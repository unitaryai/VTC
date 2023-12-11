import glob
import html
import json
import os

import pandas as pd
from google.cloud import translate_v2 as translate

# This is the video mapping from
# "Response to LiveBot: Generating Live Video Comments Based on Visual and Textual Contexts"
# https://github.com/fireflyHunter/OpenNMT-Livebot
# Which fixes several issues with the original. The ids differ from the original,
# since the original does not give the mapping of their ids to the raw videos.
VIDEO_MAPPING = "video_map_opennmt_livebot.json"

VIDEO_DIR = "~/Code/livebot/data/video/video"

translate_client = translate.Client()

os.makedirs("translated_titles", exist_ok=True)

# Translate titles
with open(VIDEO_MAPPING, "r") as f:
    video_map = json.load(f)

id_to_title = {v: k for k, v in video_map.items()}

for title, vid_id in video_map.items():
    out_file = f"translated_titles/{vid_id}.json"

    if os.path.exists(out_file):
        print("Exists", vid_id)
        continue

    print(vid_id, title)

    title_tr = translate_client.translate(title, target_language="en")

    print(title_tr)

    with open(out_file, "w") as fo:
        json.dump(title_tr, fo)

# Translate test set

os.makedirs("translated_comments", exist_ok=True)

with open("opennmt_livebot_split/test.json", "r") as f:
    test = json.load(f)

test_vid_ids = sorted(map(int, test.keys()))

for test_id in test_vid_ids:
    out_file = f"translated_comments/{test_id}.json"

    if os.path.exists(out_file):
        print("Exists", out_file)
        continue

    comms = test[str(test_id)]
    # Try to get a comment per second
    prev_time = -1

    comments_to_translate = []

    for comm_idx, comment in enumerate(comms):
        time = comment["time"]
        text = comment["danmu"]

        if "哈哈" in text:
            # ignore "haha" spam
            continue

        if time == prev_time:
            continue

        prev_time = time

        comments_to_translate.append((comm_idx, comment))

        if len(comments_to_translate) >= 5:
            break

    translated_comments = []
    for comm_idx, comment in comments_to_translate:
        comm_tr = translate_client.translate(comment["danmu"], target_language="en")
        comment["original_index"] = comm_idx
        comment["translate_json"] = comm_tr
        translated_comments.append(comment)

    print(translated_comments)

    with open(out_file, "w") as fo:
        json.dump(translated_comments, fo)

all_flvs = glob.glob(f"{VIDEO_DIR}/**/*.flv", recursive=True)

test_rows = []
for test_id in test_vid_ids:
    with open(f"translated_titles/{test_id}.json", "r") as f:
        title_json = json.load(f)
    with open(f"translated_comments/{test_id}.json", "r") as f:
        comm_json = json.load(f)

    title_zh = id_to_title[test_id]

    for flv in all_flvs:
        if title_zh in flv:
            class_dir, filename = flv.split("/")[-2:]
            break
    else:  # nobreak
        raise Exception("flv not found", title_zh)

    video_path = f"{class_dir}/{filename}"

    comments_en = [
        html.unescape(x["translate_json"]["translatedText"]) for x in comm_json
    ]
    comments_zh = [x["danmu"] for x in comm_json]

    row = {
        "id_opennmt_livebot": test_id,
        "title": html.unescape(title_json["translatedText"]),
        "title_zh": title_zh,
        "video_path": video_path,
        "comments": comments_en,
        "comments_zh": comments_zh,
    }

    test_rows.append(row)

df = pd.DataFrame(test_rows)

df.to_csv("livebot_test_translated_5comms.csv")
