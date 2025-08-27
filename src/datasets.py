import os
import json
from typing import Tuple, Optional, List

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import open_clip

DATA_DIR_DEFAULT = "/kaggle/input/facebook-hateful-meme-dataset"
DATA_SUBDIR = "data"  # Matches your notebook


class HatefulMemesDataset(Dataset):
    def __init__(self, jsonl_path: str, img_dir: str, image_size: int = 224, is_train: bool = False) -> None:
        self.img_dir = img_dir
        self.samples: List[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                img_path = os.path.join(img_dir, data["img"])
                text = data["text"]
                label = data.get("label")
                if label is not None:
                    label = float(label)
                sample_id = data.get("id")
                self.samples.append({"image": img_path, "text": text, "label": label, "id": sample_id})

        # Same transforms you used around CLIP-B/32 with OpenCLIP normalization
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))
            ])

        # Tokenizer per your notebook
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        entry = self.samples[idx]
        image = Image.open(entry["image"]).convert("RGB")
        image = self.transform(image)
        text_tokens = self.tokenizer([entry["text"]])[0]  # Returns LongTensor
        label = entry["label"]
        sample_id = entry["id"]
        return image, text_tokens, label, sample_id


def make_paths(data_dir_root: str) -> Tuple[str, str, str]:
    data_dir = os.path.join(data_dir_root, DATA_SUBDIR)
    images_dir = data_dir  # Your notebook sets images_dir = data_dir
    train_path = os.path.join(data_dir, "train.jsonl")
    dev_path = os.path.join(data_dir, "dev.jsonl")
    return images_dir, train_path, dev_path
