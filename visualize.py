# visualize_detections.py

import os, json, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor, Normalize
from transformers import ViTForImageClassification
import random

# Detector & wrapper (as before)...
class FeatureSqueezeDetector(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def score(self, x):
        x_s = torch.floor(x * 15.0) / 15.0
        p = F.softmax(self.model(x),   dim=-1)
        q = F.softmax(self.model(x_s), dim=-1)
        return (p-q).abs().sum(dim=1)

class AutoencoderDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,16,3,2,1), nn.ReLU(),
            nn.Conv2d(16,8,3,2,1),  nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8,16,3,2,1,1), nn.ReLU(),
            nn.ConvTranspose2d(16,3,3,2,1,1),  nn.Tanh()
        )
    def forward(self, x): return self.decoder(self.encoder(x))
    def score(self, x):
        recon = self(x)
        return ((x-recon)**2).mean(dim=[1,2,3])

class ViTWrapper(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(pixel_values=x).logits

def unnormalize(t):
    return (t*0.5 + 0.5).permute(1,2,0).cpu().numpy()

def main():
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Test‐set transform & load originals
    transform = Compose([Resize(224), Grayscale(3), ToTensor(),
                         Normalize([0.5]*3, [0.5]*3)])
    test_ds = MNIST("data", train=False, download=True, transform=transform)

    # 2) Load ALL FGSM adv examples to CPU only
    X_fgsm, Y_fgsm = torch.load("checkpoints/adv_fgsm.pt", map_location="cpu")

    # 3) Choose your handful of examples (first 5 here)
    sel =  random.sample(range(len(X_fgsm)), 5) # list(range(5)) 

    # 4) Move just those to GPU
    X_sel = X_fgsm[sel].to(device)
    Y_sel = Y_fgsm[sel].to(device)

    # 5) Load thresholds
    ae_delta = json.load(open("checkpoints/threshold.json"))["recon_error_threshold"]
    fs_tau   = json.load(open("checkpoints/fs_threshold.json"))["fs_threshold"]

    # 6) Instantiate ViT + detectors on GPU
    hf = ViTForImageClassification.from_pretrained(
         "farleyknight-org-username/vit-base-mnist"
    ).to(device).eval()
    vit = ViTWrapper(hf).to(device).eval()
    fs = FeatureSqueezeDetector(vit).to(device).eval()

    ae = AutoencoderDetector().to(device).eval()
    ae.load_state_dict(torch.load("checkpoints/autoencoder.pth",
                                  map_location=device))

    # 7) Score only our selected examples
    with torch.no_grad():
        fs_scores = fs.score(X_sel)
        ae_scores = ae.score(X_sel)

    # 8) Plot and save
    fig, axes = plt.subplots(len(sel), 2, figsize=(6, 3*len(sel)))
    for i, idx in enumerate(sel):
        orig, _ = test_ds[idx]
        adv = X_sel[i].cpu()

        # Original
        axes[i,0].imshow(unnormalize(orig), cmap="gray")
        axes[i,0].set_title(f"Orig #{idx}")
        axes[i,0].axis("off")

        # Adversarial
        axes[i,1].imshow(unnormalize(adv), cmap="gray")
        axes[i,1].set_title(
            f"FGSM #{idx}\nFS={fs_scores[i]:.3f}, AE={ae_scores[i]:.3f}"
        )
        axes[i,1].axis("off")

    plt.tight_layout()
    out = "results/detections_random.png"
    plt.savefig(out)
    print(f"✅ Visualization saved to {out}")

if __name__=="__main__":
    main()
