import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import ViTForImageClassification
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

from utils.data import get_dataloaders

# --- Detector definitions ---

class FeatureSqueezeDetector(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def score(self, x):
        x_s = torch.floor(x * 15.0) / 15.0
        p = F.softmax(self.model(x), dim=-1)
        q = F.softmax(self.model(x_s), dim=-1)
        return (p - q).abs().sum(dim=1)

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

# --- Helper to collect scores ---

def collect_scores(detector, loader, device):
    detector.to(device).eval()
    all_scores = []
    with torch.no_grad():
        for x, _ in tqdm(loader, desc=f"Scoring {detector.__class__.__name__}"):
            x = x.to(device)
            scores = detector.score(x)
            all_scores.append(scores.cpu().numpy())
    return np.concatenate(all_scores)

# --- Main evaluation ---

def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluating on {device}")

    # Load ViT MNIST model
    hf_model = ViTForImageClassification.from_pretrained(
        "farleyknight-org-username/vit-base-mnist"
    ).to(device).eval()
    class ViTWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(pixel_values=x).logits
    model = ViTWrapper(hf_model).to(device).eval()

    # Load autoencoder and its threshold (δ)
    ae = AutoencoderDetector().to(device)
    ae.load_state_dict(torch.load("checkpoints/autoencoder.pth", map_location=device))
    delta_threshold = json.load(open("checkpoints/threshold.json"))['recon_error_threshold']
    print(f"[INFO] AE δ threshold: {delta_threshold:.6e}")

    # Load Feature-Squeeze threshold (τ)
    fs_threshold = json.load(open("checkpoints/fs_threshold.json"))['fs_threshold']
    print(f"[INFO] FS τ threshold: {fs_threshold:.6e}")

    # Instantiate detectors
    fs_detector = FeatureSqueezeDetector(model)
    ae_detector = ae

    # Load datasets
    _, _, test_loader = get_dataloaders(
        img_size=224,
        batch_size_train=64,
        batch_size_test=256,
        val_split=0.1,
        seed=42,
        num_workers=0,
        pin_memory=False
    )

    # Load adversarial datasets
    X_fgsm, Y_fgsm = torch.load("checkpoints/adv_fgsm.pt")
    X_pgd,  Y_pgd  = torch.load("checkpoints/adv_pgd.pt")
    fgsm_loader = DataLoader(TensorDataset(X_fgsm, Y_fgsm), batch_size=256, shuffle=False)
    pgd_loader  = DataLoader(TensorDataset(X_pgd,  Y_pgd),  batch_size=256, shuffle=False)
    print(f"[INFO] Loaded adv sets: FGSM({len(X_fgsm)}), PGD({len(X_pgd)})")

    # Collect scores on clean and adversarial sets
    fs_clean = collect_scores(fs_detector, test_loader, device)
    fs_fgsm  = collect_scores(fs_detector, fgsm_loader,  device)
    fs_pgd   = collect_scores(fs_detector, pgd_loader,   device)
    ae_clean = collect_scores(ae_detector, test_loader, device)
    ae_fgsm  = collect_scores(ae_detector, fgsm_loader,  device)
    ae_pgd   = collect_scores(ae_detector, pgd_loader,   device)

    # Prepare labels
    y_clean = np.zeros_like(fs_clean)
    y_adv   = np.ones_like (fs_fgsm)
    y_true  = np.concatenate([y_clean, y_adv, y_adv])
    fs_scores = np.concatenate([fs_clean, fs_fgsm, fs_pgd])
    ae_scores = np.concatenate([ae_clean, ae_fgsm, ae_pgd])

    # Compute ROC metrics
    fpr_fs, tpr_fs, _ = roc_curve(y_true, fs_scores, pos_label=1)
    roc_auc_fs        = auc(fpr_fs, tpr_fs)
    fpr_ae, tpr_ae, _ = roc_curve(y_true, ae_scores, pos_label=1)
    roc_auc_ae        = auc(fpr_ae, tpr_ae)

    # Compute Precision-Recall metrics
    prec_fs, rec_fs, _ = precision_recall_curve(y_true, fs_scores, pos_label=1)
    ap_fs              = auc(rec_fs, prec_fs)
    prec_ae, rec_ae, _ = precision_recall_curve(y_true, ae_scores, pos_label=1)
    ap_ae              = auc(rec_ae, prec_ae)

    # Ensure results directory
    os.makedirs("results", exist_ok=True)

    # Plot & save ROC curve
    plt.figure(figsize=(6,4))
    plt.plot(fpr_fs, tpr_fs, label=f'FS (AUC={roc_auc_fs:.2f})')
    plt.plot(fpr_ae, tpr_ae, label=f'AE (AUC={roc_auc_ae:.2f})')
    plt.plot([0,1],[0,1],'k--',alpha=0.3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('results/roc_curve.png')
    plt.close()

    # Plot & save Precision-Recall curve
    plt.figure(figsize=(6,4))
    plt.plot(rec_fs, prec_fs, label=f'FS (AP={ap_fs:.2f})')
    plt.plot(rec_ae, prec_ae, label=f'AE (AP={ap_ae:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision–Recall Curves')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('results/pr_curve.png')
    plt.close()

    # Compute detection rates at thresholds
    fs_far_clean = float((fs_clean  > fs_threshold ).mean())
    fs_dr_fgsm   = float((fs_fgsm   > fs_threshold ).mean())
    fs_dr_pgd    = float((fs_pgd    > fs_threshold ).mean())
    ae_far_clean = float((ae_clean  > delta_threshold).mean())
    ae_dr_fgsm   = float((ae_fgsm   > delta_threshold).mean())
    ae_dr_pgd    = float((ae_pgd    > delta_threshold).mean())

    # Save numeric metrics
    metrics = {
        "thresholds": {
            "ae_delta": delta_threshold,
            "fs_tau":  fs_threshold
        },
        "roc_auc": {
            "feature_squeeze": roc_auc_fs,
            "autoencoder":      roc_auc_ae
        },
        "average_precision": {
            "feature_squeeze": ap_fs,
            "autoencoder":      ap_ae
        },
        "detection_at_thresholds": {
            "feature_squeeze": {
                "FAR_clean": fs_far_clean,
                "DR_FGSM":   fs_dr_fgsm,
                "DR_PGD":    fs_dr_pgd
            },
            "autoencoder": {
                "FAR_clean": ae_far_clean,
                "DR_FGSM":   ae_dr_fgsm,
                "DR_PGD":    ae_dr_pgd
            }
        }
    }
    with open('results/metrics.json', 'w') as fp:
        json.dump(metrics, fp, indent=2)

    print("✅ Saved ROC & PR plots and metrics in results/")

if __name__ == '__main__':
    main()
