# HatefulMeme-CLIPper (Kaggle CLI)

Implements your Kaggle notebook flow as scripts:
- OpenCLIP ViT-B/32 ("openai") encoders
- Freeze all params except last visual transformer block
- Project image/text → outer-product → MLP → BCEWithLogits
- AMP + early stopping by AUROC
- Dataset: `parthplc/facebook-hateful-meme-dataset` → `/kaggle/input/facebook-hateful-meme-dataset/data/...`

## 0) Clone
git clone https://github.com/<you>/hatefulmeme-clipper.git
cd hatefulmeme-clipper

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
