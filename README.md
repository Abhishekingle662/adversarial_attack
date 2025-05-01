# Adversarial Attack Detection on MNIST with ViT

This project implements and evaluates two simple detectors  
– Feature Squeeze and Autoencoder reconstruction error –  
against adversarial MNIST examples crafted on a pre-trained ViT model.

## Directory structure

- `train_autoencoder.py`  
  Train an image autoencoder on MNIST and compute a reconstruction‐error threshold δ.

- `make_attacks.py`  
  Generate FGSM and PGD adversarial examples against the ViT MNIST classifier.

- `eval_detectors.py`  
  Load thresholds and adversarial sets, score clean/adversarial inputs,  
  and compute ROC & PR curves and detection rates at thresholds.

- `visualize.py`  
  (Optional) Visualize individual scores/reconstructions.

- `utils/data.py`  
  Data loading utilities for MNIST (train/val/test splits).

- `checkpoints/`  
  Saved autoencoder weights, thresholds, and adversarial datasets.

- `results/`  
  Generated ROC/PR plots and numeric metrics.

## Setup

1. Create & activate a Python 3.8+ environment (ideally with CUDA support):

   ```powershell
    conda create --name gpu_env python=3.8
    conda activate gpu_env
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

## Usage

1. Train the autoencoder and calibrate δ (95th percentile recon error):

   ```powershell
   python train_autoencoder.py
   ```

2. Craft adversarial examples (FGSM & PGD) and save to `checkpoints/`:

   ```powershell
   python make_attacks.py
   ```

3. Evaluate both detectors on clean & adversarial data; saves ROC/PR curves and metrics:

   ```powershell
   python eval_detectors.py
   ```

4. (Optional) Visualize reconstructions & detector scores:

   ```powershell
   python visualize.py
   ```

## Output

- `checkpoints/autoencoder.pth` & `threshold.json` (δ)  
- `checkpoints/fs_threshold.json` (τ)  
- `checkpoints/adv_fgsm.pt`, `checkpoints/adv_pgd.pt`  
- `results/roc_curve.png`, `results/pr_curve.png`, `results/metrics.json`

## License

MIT
