# Facebook Hateful Meme Finetune CLIP (Hate-CLIPper)

    Facebook_Hateful_Meme_Finetune_CLIP_Hate_CLIPper/
    ├─ env/
    │  └─ install.sh
    ├─ data/
    │  └─ download.sh
    ├─ src/
    │  ├─ datasets.py
    │  ├─ modeling.py
    │  ├─ train.py
    │  └─ infer.py
    ├─ requirements.txt
    └─ README.md

## 0) Clone

    git clone https://github.com/kanru-wang/Facebook_Hateful_Meme_Finetune_CLIP_Hate_CLIPper.git
    cd Facebook_Hateful_Meme_Finetune_CLIP_Hate_CLIPper

## 1) (Optional) Install
    bash env/install.sh

## 2) Data on Kaggle
Attach **parthplc/facebook-hateful-meme-dataset** to the notebook/script.
It appears at `/kaggle/input/facebook-hateful-meme-dataset/` and you write to `/kaggle/working`.  
(“input” is read-only; “working” is read-write.) 

## 3) Train
    python -m src.train \
      --data_dir_root /kaggle/input/facebook-hateful-meme-dataset \
      --output_dir /kaggle/working/ckpts \
      --batch_size 16 \
      --epochs 10 \
      --patience 3 \
      --lr 1e-4 \
      --weight_decay 1e-4 \
      --fp16

Artifacts:
- `/kaggle/working/ckpts/pytorch_model_best.bin`
- `/kaggle/working/ckpts/pytorch_model_final.bin`

## 4) Inference
    python -m src.infer \
      --data_dir_root /kaggle/input/facebook-hateful-meme-dataset \
      --checkpoint_path /kaggle/working/ckpts/pytorch_model_best.bin \
      --split dev \
      --batch_size 64 \
      --out_csv /kaggle/working/preds.csv
