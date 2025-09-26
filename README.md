# Facebook Hateful Meme Finetune CLIP (Hate-CLIPper)

- Hate-CLIPper: Multimodal Hateful Meme Classification based on Cross-modal Interaction of CLIP Features https://arxiv.org/pdf/2210.05916v3
- <img src="https://raw.githubusercontent.com/gokulkarthik/hateclipper/refs/heads/main/images/hateclipper.png" width="800"/>
- The projection layers is to achieve better alignment between the text and image space, i.e. finetune for this specific dataset
- The FIM representation is obtained by computing the outer product of pi and pt , i.e., R = pi ⊗pt.
- The FIM can be flattened to get a vector r of length n**2 and passed through a learnable neural network classifier to obtain the final classification decision. We refer to the fusion of text and image features using the FIM as cross-fusion.
- The sum of these diagonal elements is the dot product between pi and pt, which intuitively measures the alignment (angle) between the two vectors. We refer to the fusion of text and image features using only the diagonal elements of FIM as align-fusion.
- For low computational resource conditions, it would be appropriate to replace cross-fusion with align-fusion in the Hate-CLIPper framework.


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
  --num_workers 2 \
  --fp16
```

### Local

`python ./src/train.py --data_dir_root ./ --output_dir ./outputs --batch_size 16 --epochs 10  --patience 3 --lr 1e-4
--weight_decay 1e-4 --num_workers 2 --fp16`

fp16 is optional; use if you have GPU with CUDA

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
  --out_csv /kaggle/working/preds.csv \
  --num_workers 2
```

### Local

`python ./src/infer.py --data_dir_root ./ --checkpoint_path ./outputs/pytorch_model_best.bin --split dev --batch_size 64 --out_csv ./outputs/preds.csv --num_workers 2`

This produces a CSV file with:
- `id`
- `prob_hate` (predicted probability)
- `pred_label` (0 or 1)
