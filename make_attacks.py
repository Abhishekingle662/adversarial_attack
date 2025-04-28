# make_attacks.py

import os
import torch
import torch.nn as nn
from utils.data import get_dataloaders
from transformers import ViTForImageClassification
from torchattacks import FGSM, PGD
from tqdm import tqdm

def main():
    # 0. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Generating adversarial examples on {device}")

    # 1. Load test data with small eval batch
    _, _, test_loader = get_dataloaders(
        img_size=224,
        batch_size_train=64,
        batch_size_test=16,    # small batch for memory safety
        val_split=0.1,
        seed=42,
        num_workers=0,
        pin_memory=False
    )
    print(f"[INFO] Test set size: {len(test_loader.dataset)} samples")

    # 2. Load & wrap ViT MNIST
    hf_model = ViTForImageClassification.from_pretrained(
        "farleyknight-org-username/vit-base-mnist"
    ).to(device).eval()

    class ViTWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(pixel_values=x).logits

    model = ViTWrapper(hf_model).to(device).eval()

    # Warm up one batch (compiles CUDA kernels)
    x0, y0 = next(iter(test_loader))
    _ = model(x0.to(device))

    # 3. Configure attacks (normalized-space ε)
    eps_norm   = (8/255)  / 0.5
    alpha_norm = (2/255)  / 0.5
    fgsm = FGSM(model, eps=eps_norm)
    pgd  = PGD(model, eps=eps_norm, alpha=alpha_norm,
               steps=10, random_start=True)

    # 4. Generate FGSM examples
    X_fgsm, Y_fgsm = [], []
    print("[INFO] Crafting FGSM examples...")
    for x, y in tqdm(test_loader, desc="FGSM"):
        x, y = x.to(device), y.to(device)
        x_f = fgsm(x, y)
        X_fgsm.append(x_f.cpu())
        Y_fgsm.append(y.cpu())
        torch.cuda.empty_cache()

    # 5. Generate PGD examples
    X_pgd, Y_pgd = [], []
    print("[INFO] Crafting PGD examples...")
    for x, y in tqdm(test_loader, desc="PGD"):
        x, y = x.to(device), y.to(device)
        x_p = pgd(x, y)
        X_pgd.append(x_p.cpu())
        Y_pgd.append(y.cpu())
        torch.cuda.empty_cache()

    # 6. Save to disk
    os.makedirs("checkpoints", exist_ok=True)
    torch.save((torch.cat(X_fgsm), torch.cat(Y_fgsm)), "checkpoints/adv_fgsm.pt")
    torch.save((torch.cat(X_pgd),  torch.cat(Y_pgd )), "checkpoints/adv_pgd.pt")
    print("[DONE] Adversarial datasets saved in checkpoints/")

if __name__ == "__main__":
    main()
