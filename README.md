# Facebook Hateful Meme Finetune CLIP (Hate-CLIPper)

    Facebook_Hateful_Meme_Finetune_CLIP_Hate_CLIPper/
    ├─ src/
    |  ├─ __init__.py
    │  ├─ datasets.py
    │  ├─ modeling.py
    │  ├─ train.py
    │  └─ infer.py
    ├─ requirements.txt
    └─ README.md

## 0) Clone

    git clone https://github.com/kanru-wang/Facebook_Hateful_Meme_Finetune_CLIP_Hate_CLIPper.git

## 1) Install requirements.txt

    cd Facebook_Hateful_Meme_Finetune_CLIP_Hate_CLIPper && pip install -r requirements.txt

## 2) Data

### On Kaggle
Attach the dataset `parthplc/facebook-hateful-meme-dataset` in the right panel of your Kaggle notebook or script.  
It will be mounted under:
```
/kaggle/input/facebook-hateful-meme-dataset
```

###  Locally
1. Go to the Kaggle dataset page:  
   https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset
2. Manually download the dataset zip.
3. Unzip it into a folder named `./data` inside this repo, so you get:
```
./data/train.jsonl
./data/dev.jsonl
./data/img/...
```

## 3) Train

Run the training script

### Kaggle
```bash
!python /kaggle/working/Facebook_Hateful_Meme_Finetune_CLIP_Hate_CLIPper/src/train.py \
  --data_dir_root /kaggle/input/facebook-hateful-meme-dataset \
  --output_dir /kaggle/working/checkpoints \
  --batch_size 16 \
  --epochs 10 \
  --patience 3 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --fp16
```

### Local
```bash
python Facebook_Hateful_Meme_Finetune_CLIP_Hate_CLIPper/src/train.py \
  --data_dir_root ./data \
  --output_dir ./outputs \
  --batch_size 16 \
  --epochs 10 \
  --patience 3 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --fp16  # Optional: use if you have GPU with CUDA
```

Artifacts:
- `pytorch_model_best.bin`
- `pytorch_model_final.bin`

---

## 4) Inference

Run the inference script

### Kaggle
```bash
!python /kaggle/working/Facebook_Hateful_Meme_Finetune_CLIP_Hate_CLIPper/src/infer.py \
  --data_dir_root /kaggle/input/facebook-hateful-meme-dataset \
  --checkpoint_path /kaggle/working/checkpoints/pytorch_model_best.bin \
  --split dev \
  --batch_size 64 \
  --out_csv /kaggle/working/preds.csv
```

### Local
```bash
python Facebook_Hateful_Meme_Finetune_CLIP_Hate_CLIPper/src/infer.py \
  --data_dir_root ./data \
  --checkpoint_path ./outputs/pytorch_model_best.bin \
  --split dev \
  --batch_size 64 \
  --out_csv ./outputs/preds.csv
```

This produces a CSV file with:
- `id`
- `prob_hate` (predicted probability)
- `pred_label` (0 or 1)
