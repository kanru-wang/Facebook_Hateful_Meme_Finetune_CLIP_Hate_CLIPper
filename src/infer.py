import os, sys
import argparse
import platform
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import open_clip

sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Add repo root to path

from src.datasets import HatefulMemesDataset, make_paths, DATA_DIR_DEFAULT
from src.modeling import HateCLIPMultimodalModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference for Hate-CLIPper (your notebook logic)")
    p.add_argument("--data_dir_root", type=str, default=DATA_DIR_DEFAULT)
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--split", type=str, default="dev", choices=["train", "dev"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out_csv", type=str, default="/kaggle/working/preds.csv")
    p.add_argument("--num_workers", type=int, default=0 if platform.system() == "Windows" else 2)
    return p.parse_args()


def collate_fn(batch):
    imgs, text_tokens, labels, ids = zip(*batch)
    imgs = torch.stack(imgs)
    text_tokens = torch.stack(text_tokens)
    return imgs, text_tokens, list(ids)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images_dir, train_path, dev_path = make_paths(args.data_dir_root)
    jsonl_path = train_path if args.split == "train" else dev_path

    ds = HatefulMemesDataset(jsonl_path, images_dir, image_size=224, is_train=False)

    pin = torch.cuda.is_available()
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin,
    )

    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained="openai",
        device=device,
        # force QuickGELU if checkpoint expects it
        quick_gelu=True
    )
    model = HateCLIPMultimodalModel(clip_model).to(device)

    state = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    all_logits = []
    all_ids = []
    for images, text_tokens, ids in loader:
        images = images.to(device)
        text_tokens = text_tokens.to(device)
        logits = model(images, text_tokens)
        all_logits.append(logits.detach().cpu().numpy())
        all_ids.extend(ids)

    logits = np.concatenate(all_logits)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    df = pd.DataFrame({"id": all_ids, "prob_hate": probs, "pred_label": preds})
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
