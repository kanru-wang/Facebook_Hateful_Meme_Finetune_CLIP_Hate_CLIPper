import os, sys
import argparse
from typing import Dict, Tuple

import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import open_clip
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Add repo root to path

from src.datasets import HatefulMemesDataset, make_paths, DATA_DIR_DEFAULT
from src.modeling import HateCLIPMultimodalModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finetune Hate-CLIPper (your notebook logic) on Hateful Memes")
    p.add_argument("--data_dir_root", type=str, default=DATA_DIR_DEFAULT)
    p.add_argument("--output_dir", type=str, default="/kaggle/working/checkpoints")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def seed_everything(seed: int) -> None:
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    for images, text_tokens, labels, _ in loader:
        images = images.to(device)
        text_tokens = text_tokens.to(device)
        if labels is not None:
            labels = torch.tensor(labels, dtype=torch.float32).to(device)
        logits = model(images, text_tokens)
        if labels is not None:
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_logits.extend(logits.detach().cpu().numpy().tolist())

    avg_loss = (total_loss / len(loader.dataset)) if all_labels else None
    metrics: Dict[str, float] = {}
    if all_labels:
        probs = 1.0 / (1.0 + np.exp(-np.array(all_logits)))
        metrics["auroc"] = float(roc_auc_score(all_labels, probs))
        preds = (probs >= 0.5).astype(int)
        metrics["accuracy"] = float(accuracy_score(all_labels, preds))
    model.train()
    return avg_loss, metrics


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    images_dir, train_path, dev_path = make_paths(args.data_dir_root)

    # Datasets and loaders (same transforms and tokenizer as your NB)
    train_ds = HatefulMemesDataset(train_path, images_dir, image_size=224, is_train=True)
    val_ds = HatefulMemesDataset(dev_path, images_dir, image_size=224, is_train=False)

    def collate_fn(batch):
        imgs, text_tokens, labels, ids = zip(*batch)
        imgs = torch.stack(imgs)
        text_tokens = torch.stack(text_tokens)  # LongTensor [B, T]
        labels_tensor = None if labels[0] is None else torch.tensor(labels, dtype=torch.float32)
        return imgs, text_tokens, labels_tensor, list(ids)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # OpenCLIP ViT-B/32 "openai"
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = HateCLIPMultimodalModel(clip_model).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler() if args.fp16 and device.type == "cuda" else None

    best_auroc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for images, text_tokens, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images = images.to(device)
            text_tokens = text_tokens.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            if scaler is not None:
                with autocast():
                    logits = model(images, text_tokens)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images, text_tokens)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * labels.size(0)

        avg_train = train_loss / len(train_loader.dataset)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        auroc = val_metrics.get("auroc", 0.0)
        acc = val_metrics.get("accuracy", 0.0)
        print(
            f"Epoch {epoch}: Train Loss = {avg_train:.4f}, Val Loss = {val_loss:.4f}, Val AUROC = {auroc:.4f}, Val Acc = {acc:.4f}")

        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print("Early stopping")
                break

    # Save final and best
    final_path = os.path.join(args.output_dir, "pytorch_model_final.bin")
    torch.save(model.state_dict(), final_path)
    print(f"Saved: {final_path}")
    if best_state is not None:
        best_path = os.path.join(args.output_dir, "pytorch_model_best.bin")
        torch.save(best_state, best_path)
        print(f"Saved: {best_path}")


if __name__ == "__main__":
    main()
