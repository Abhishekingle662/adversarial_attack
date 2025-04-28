# train_autoencoder.py

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data import get_dataloaders

def main():
    # 0. Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1. Load data (train + val split)
    print("[INFO] Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        img_size=224,
        batch_size_train=64,
        batch_size_test=256,
        val_split=0.1,
        seed=42,
        num_workers=4,
        pin_memory=True
    )
    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    n_test  = len(test_loader.dataset)
    print(f"[INFO] Dataset sizes — train: {n_train}, val: {n_val}, test: {n_test}")

    # 2. Define the Autoencoder
    class AutoencoderDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(16,  8, 3, 2, 1), nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(8, 16, 3, 2, 1, 1), nn.ReLU(),
                nn.ConvTranspose2d(16, 3,  3, 2, 1, 1), nn.Tanh(),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

        def score(self, x):
            recon = self(x)
            return ((x - recon) ** 2).mean(dim=[1,2,3])

    # 3. Instantiate model & optimizer
    ae = AutoencoderDetector().to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    print("[INFO] Autoencoder initialized:")
    print(ae)

    # 4. Training loop
    num_epochs = 10
    print(f"[INFO] Starting training for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        ae.train()
        running_loss = 0.0

        for batch_idx, (x, _) in enumerate(train_loader, start=1):
            x = x.to(device)
            recon = ae(x)
            loss = F.mse_loss(recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

            if batch_idx % 100 == 0 or batch_idx == len(train_loader):
                print(f"  [Epoch {epoch}] Batch {batch_idx}/{len(train_loader)} — Batch MSE: {loss.item():.6f}")

        train_loss = running_loss / n_train
        epoch_time = time.time() - epoch_start

        # compute val MSE
        ae.eval()
        val_errors = []
        with torch.no_grad():
            for x_val, _ in val_loader:
                x_val = x_val.to(device)
                errs = ((ae(x_val) - x_val) ** 2).mean(dim=[1,2,3])
                val_errors.append(errs.cpu().numpy())
        val_errors = np.concatenate(val_errors)
        val_loss = val_errors.mean()

        print(f"[RESULT] Epoch {epoch} completed in {epoch_time:.1f}s — Train MSE: {train_loss:.6f} — Val MSE: {val_loss:.6f}")

    # 5. Threshold calibration
    threshold = float(np.percentile(val_errors, 95))
    print(f"[INFO] Computed 95th-percentile recon-error threshold: {threshold:.6f}")

    # 6. Save outputs
    os.makedirs("checkpoints", exist_ok=True)
    model_path = "checkpoints/autoencoder.pth"
    th_path    = "checkpoints/threshold.json"
    torch.save(ae.state_dict(), model_path)
    with open(th_path, "w") as fp:
        json.dump({"recon_error_threshold": threshold}, fp)
    print(f"[INFO] Saved weights to {model_path}")
    print(f"[INFO] Saved threshold to {th_path}")
    print("[DONE] Autoencoder training and threshold calibration complete.")

        # ===== Calibrate Feature-Squeeze Threshold =====
    from transformers import ViTForImageClassification

    # 1) Reload ViT and wrap
    hf_model = ViTForImageClassification.from_pretrained(
        "farleyknight-org-username/vit-base-mnist"
    ).to(device).eval()

    class ViTWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(pixel_values=x).logits

    model = ViTWrapper(hf_model).to(device).eval()

    # 2) Instantiate FS detector
    from eval_detectors import FeatureSqueezeDetector
    fs_detector = FeatureSqueezeDetector(model)

    # 3) Score val set
    fs_detector.eval()
    fs_val_scores = []
    with torch.no_grad():
        for x_val, _ in val_loader:
            x_val = x_val.to(device)
            s = fs_detector.score(x_val)
            fs_val_scores.append(s.cpu().numpy())
    fs_val_scores = np.concatenate(fs_val_scores)

    # 4) Compute and save τ
    fs_threshold = float(np.percentile(fs_val_scores, 95))
    print(f"[INFO] FS threshold (95th pct val): {fs_threshold:.6e}")

    fs_path = "checkpoints/fs_threshold.json"
    with open(fs_path, "w") as fp:
        json.dump({"fs_threshold": fs_threshold}, fp)
    print(f"[INFO] Saved FS threshold to {fs_path}")


if __name__ == "__main__":
    # Necessary for Windows multiprocessing with spawn
    torch.multiprocessing.freeze_support()
    main()
