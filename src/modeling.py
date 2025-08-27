from typing import Tuple

import torch
import torch.nn as nn


class HateCLIPMultimodalModel(nn.Module):
    def __init__(self, clip_model: nn.Module, image_dim: int = 512, text_dim: int = 512,
                 proj_dim: int = 512, hidden_dim: int = 128) -> None:
        super().__init__()
        self.clip_model = clip_model

        # Freeze all CLIP params, unfreeze the last visual transformer block — same as your notebook
        for name, param in self.clip_model.named_parameters():
            param.requires_grad = False
            if "visual.transformer.resblocks.11" in name:
                param.requires_grad = True

        self.image_proj = nn.Linear(image_dim, proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim)

        # Your notebook shows an MLP on top of the flattened outer product.
        # If your hidden/dropout differ, you can tweak here to match 1:1.
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * proj_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        # Keep CLIP encoders in no_grad() per your training loop (only the chosen block has grads)
        with torch.no_grad():
            img_features = self.clip_model.encode_image(images)  # [B, 512] for ViT-B/32
            text_features = self.clip_model.encode_text(text_tokens)  # [B, 512]

        p_i = self.image_proj(img_features)  # [B, 512]
        p_t = self.text_proj(text_features)  # [B, 512]

        outer = torch.einsum("bi,bj->bij", p_i, p_t)  # [B, 512, 512]
        r = outer.view(outer.size(0), -1)  # [B, 512*512]
        logit = self.classifier(r).squeeze(1)  # [B]
        return logit
