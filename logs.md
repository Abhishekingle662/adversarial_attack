# Train_Autoencoder.py file logs and outputs:

[INFO] Using device: cuda
[INFO] Loading data...
[INFO] Dataset sizes — train: 54000, val: 6000, test: 10000
[INFO] Autoencoder initialized:
AutoencoderDetector(
  (encoder): Sequential(
    (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(16, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (3): ReLU()
  )
  (decoder): Sequential(
    (0): ConvTranspose2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (1): ReLU()
    (2): ConvTranspose2d(16, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (3): Tanh()
  )
)
[INFO] Starting training for 10 epochs...
  [Epoch 1] Batch 100/844 — Batch MSE: 0.015504
  [Epoch 1] Batch 200/844 — Batch MSE: 0.006098
  [Epoch 1] Batch 300/844 — Batch MSE: 0.004256
  [Epoch 1] Batch 400/844 — Batch MSE: 0.002869
  [Epoch 1] Batch 500/844 — Batch MSE: 0.002225
  [Epoch 1] Batch 600/844 — Batch MSE: 0.001618
  [Epoch 1] Batch 700/844 — Batch MSE: 0.001340
  [Epoch 1] Batch 800/844 — Batch MSE: 0.001108
  [Epoch 1] Batch 844/844 — Batch MSE: 0.001023
[RESULT] Epoch 1 completed in 80.7s — Train MSE: 0.032365 — Val MSE: 0.001002
  [Epoch 2] Batch 100/844 — Batch MSE: 0.000829
  [Epoch 2] Batch 200/844 — Batch MSE: 0.000716
  [Epoch 2] Batch 300/844 — Batch MSE: 0.000603
  [Epoch 2] Batch 400/844 — Batch MSE: 0.000550
  [Epoch 2] Batch 500/844 — Batch MSE: 0.000476
  [Epoch 2] Batch 600/844 — Batch MSE: 0.000468
  [Epoch 2] Batch 700/844 — Batch MSE: 0.000411
  [Epoch 2] Batch 800/844 — Batch MSE: 0.000378
  [Epoch 2] Batch 844/844 — Batch MSE: 0.000361
[RESULT] Epoch 2 completed in 81.9s — Train MSE: 0.000584 — Val MSE: 0.000370
  [Epoch 3] Batch 100/844 — Batch MSE: 0.000347
  [Epoch 3] Batch 200/844 — Batch MSE: 0.000298
  [Epoch 3] Batch 300/844 — Batch MSE: 0.000274
  [Epoch 3] Batch 400/844 — Batch MSE: 0.000270
  [Epoch 3] Batch 500/844 — Batch MSE: 0.000260
  [Epoch 3] Batch 600/844 — Batch MSE: 0.000259
  [Epoch 3] Batch 700/844 — Batch MSE: 0.000226
  [Epoch 3] Batch 800/844 — Batch MSE: 0.000235
  [Epoch 3] Batch 844/844 — Batch MSE: 0.000210
[RESULT] Epoch 3 completed in 77.2s — Train MSE: 0.000279 — Val MSE: 0.000222
  [Epoch 4] Batch 100/844 — Batch MSE: 0.000212
  [Epoch 4] Batch 200/844 — Batch MSE: 0.000206
  [Epoch 4] Batch 300/844 — Batch MSE: 0.000198
  [Epoch 4] Batch 400/844 — Batch MSE: 0.000181
  [Epoch 4] Batch 500/844 — Batch MSE: 0.000188
  [Epoch 4] Batch 600/844 — Batch MSE: 0.000190
  [Epoch 4] Batch 700/844 — Batch MSE: 0.000159
  [Epoch 4] Batch 800/844 — Batch MSE: 0.000164
  [Epoch 4] Batch 844/844 — Batch MSE: 0.000168
[RESULT] Epoch 4 completed in 81.3s — Train MSE: 0.000190 — Val MSE: 0.000166
  [Epoch 5] Batch 100/844 — Batch MSE: 0.000160
  [Epoch 5] Batch 200/844 — Batch MSE: 0.000155
  [Epoch 5] Batch 300/844 — Batch MSE: 0.000156
  [Epoch 5] Batch 400/844 — Batch MSE: 0.000148
  [Epoch 5] Batch 500/844 — Batch MSE: 0.000134
  [Epoch 5] Batch 600/844 — Batch MSE: 0.000138
  [Epoch 5] Batch 700/844 — Batch MSE: 0.000144
  [Epoch 5] Batch 800/844 — Batch MSE: 0.000132
  [Epoch 5] Batch 844/844 — Batch MSE: 0.000127
[RESULT] Epoch 5 completed in 84.1s — Train MSE: 0.000145 — Val MSE: 0.000127
  [Epoch 6] Batch 100/844 — Batch MSE: 0.000123
  [Epoch 6] Batch 200/844 — Batch MSE: 0.000123
  [Epoch 6] Batch 300/844 — Batch MSE: 0.000125
  [Epoch 6] Batch 400/844 — Batch MSE: 0.000116
  [Epoch 6] Batch 500/844 — Batch MSE: 0.000113
  [Epoch 6] Batch 600/844 — Batch MSE: 0.000115
  [Epoch 6] Batch 700/844 — Batch MSE: 0.000103
  [Epoch 6] Batch 800/844 — Batch MSE: 0.000101
  [Epoch 6] Batch 844/844 — Batch MSE: 0.000114
[RESULT] Epoch 6 completed in 85.3s — Train MSE: 0.000114 — Val MSE: 0.000110
  [Epoch 7] Batch 100/844 — Batch MSE: 0.000108
  [Epoch 7] Batch 200/844 — Batch MSE: 0.000093
  [Epoch 7] Batch 300/844 — Batch MSE: 0.000091
  [Epoch 7] Batch 400/844 — Batch MSE: 0.000090
  [Epoch 7] Batch 500/844 — Batch MSE: 0.000097
  [Epoch 7] Batch 600/844 — Batch MSE: 0.000083
  [Epoch 7] Batch 700/844 — Batch MSE: 0.000088
  [Epoch 7] Batch 800/844 — Batch MSE: 0.000082
  [Epoch 7] Batch 844/844 — Batch MSE: 0.000080
[RESULT] Epoch 7 completed in 80.1s — Train MSE: 0.000091 — Val MSE: 0.000080
  [Epoch 8] Batch 100/844 — Batch MSE: 0.000081
  [Epoch 8] Batch 200/844 — Batch MSE: 0.000081
  [Epoch 8] Batch 300/844 — Batch MSE: 0.000074
  [Epoch 8] Batch 400/844 — Batch MSE: 0.000080
  [Epoch 8] Batch 500/844 — Batch MSE: 0.000069
  [Epoch 8] Batch 600/844 — Batch MSE: 0.000069
  [Epoch 8] Batch 700/844 — Batch MSE: 0.000068
  [Epoch 8] Batch 800/844 — Batch MSE: 0.000067
  [Epoch 8] Batch 844/844 — Batch MSE: 0.000062
[RESULT] Epoch 8 completed in 87.0s — Train MSE: 0.000074 — Val MSE: 0.000066
  [Epoch 9] Batch 100/844 — Batch MSE: 0.000063
  [Epoch 9] Batch 200/844 — Batch MSE: 0.000061
  [Epoch 9] Batch 300/844 — Batch MSE: 0.000062
  [Epoch 9] Batch 400/844 — Batch MSE: 0.000059
  [Epoch 9] Batch 500/844 — Batch MSE: 0.000058
  [Epoch 9] Batch 600/844 — Batch MSE: 0.000061
  [Epoch 9] Batch 700/844 — Batch MSE: 0.000055
  [Epoch 9] Batch 800/844 — Batch MSE: 0.000057
  [Epoch 9] Batch 844/844 — Batch MSE: 0.000053
[RESULT] Epoch 9 completed in 83.3s — Train MSE: 0.000061 — Val MSE: 0.000055
  [Epoch 10] Batch 100/844 — Batch MSE: 0.000053
  [Epoch 10] Batch 200/844 — Batch MSE: 0.000055
  [Epoch 10] Batch 300/844 — Batch MSE: 0.000054
  [Epoch 10] Batch 400/844 — Batch MSE: 0.000053
  [Epoch 10] Batch 500/844 — Batch MSE: 0.000052
  [Epoch 10] Batch 600/844 — Batch MSE: 0.000052
  [Epoch 10] Batch 700/844 — Batch MSE: 0.000047
  [Epoch 10] Batch 800/844 — Batch MSE: 0.000051
  [Epoch 10] Batch 844/844 — Batch MSE: 0.000049
[RESULT] Epoch 10 completed in 84.9s — Train MSE: 0.000052 — Val MSE: 0.000047
[INFO] Computed 95th-percentile recon-error threshold: 0.000066
[INFO] Saved weights to checkpoints/autoencoder.pth
[INFO] Saved threshold to checkpoints/threshold.json
[DONE] Autoencoder training and threshold calibration complete.




# Make_Attacks.py logs/outputs:
[INFO] Generating adversarial examples on cuda
[INFO] Test set size: 10000 samples
C:\Users\Abhis\.conda\envs\torch_gpu_env\lib\site-packages\transformers\models\vit\modeling_vit.py:277: UserWarning: 1Torch was not compiled with flash attention. (Triggered internally at C:\cb\pytorch_1000000000000\work\aten\src\ATen\native\transformers\cuda\sdp_utils.cpp:555.)
  context_layer = torch.nn.functional.scaled_dot_product_attention(
[INFO] Crafting FGSM examples...
FGSM: 100%|███████████████████████████████████████████████████| 625/625 [04:05<00:00,  2.54it/s]
[INFO] Crafting PGD examples...
PGD: 100%|████████████████████████████████████████████████████| 625/625 [24:29<00:00,  2.35s/it]
[DONE] Saved adversarial datasets to checkpoints/



# eval_detectors.py logs/outputs:
[INFO] Generating adversarial examples on cuda
[INFO] Test set size: 10000 samples
C:\Users\Abhis\.conda\envs\torch_gpu_env\lib\site-packages\transformers\models\vit\modeling_vit.py:277: UserWarning: 1Torch was not compiled with flash attention. (Triggered internally at C:\cb\pytorch_1000000000000\work\aten\src\ATen\native\transformers\cuda\sdp_utils.cpp:555.)
  context_layer = torch.nn.functional.scaled_dot_product_attention(
[INFO] Crafting FGSM examples...
FGSM: 100%|███████████████████████████████████████████████████| 625/625 [04:05<00:00,  2.54it/s]
[INFO] Crafting PGD examples...
PGD: 100%|████████████████████████████████████████████████████| 625/625 [24:29<00:00,  2.35s/it]
[DONE] Saved adversarial datasets to checkpoints/
(torch_gpu_env) PS D:\Final_Project> python .\eval_detectors.py
[INFO] Evaluating on cuda
.\eval_detectors.py:85: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  ae.load_state_dict(torch.load(ae_ckpt, map_location=device))
[INFO] Loaded AE weights from checkpoints/autoencoder.pth
[INFO] Using AE threshold δ = 0.000066
.\eval_detectors.py:110: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  X_fgsm, Y_fgsm = torch.load(fgsm_path)
.\eval_detectors.py:111: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  X_pgd,  Y_pgd  = torch.load(pgd_path)
[INFO] Loaded adv sets: FGSM(10000), PGD(10000)
Scoring with FeatureSqueezeDetector:   0%|                               | 0/40 [00:00<?, ?it/s]C:\Users\Abhis\.conda\envs\torch_gpu_env\lib\site-packages\transformers\models\vit\modeling_vit.py:277: UserWarning: 1Torch was not compiled with flash attention. (Triggered internally at C:\cb\pytorch_1000000000000\work\aten\src\ATen\native\transformers\cuda\sdp_utils.cpp:555.)
  context_layer = torch.nn.functional.scaled_dot_product_attention(
Scoring with FeatureSqueezeDetector: 100%|██████████████████████| 40/40 [03:18<00:00,  4.97s/it]
Scoring with FeatureSqueezeDetector: 100%|██████████████████████| 40/40 [02:31<00:00,  3.78s/it]
Scoring with FeatureSqueezeDetector: 100%|██████████████████████| 40/40 [02:32<00:00,  3.80s/it]
Scoring with AutoencoderDetector: 100%|█████████████████████████| 40/40 [01:00<00:00,  1.50s/it]
Scoring with AutoencoderDetector: 100%|█████████████████████████| 40/40 [00:10<00:00,  3.91it/s]
Scoring with AutoencoderDetector: 100%|█████████████████████████| 40/40 [00:08<00:00,  4.72it/s]

[METRICS @ AE threshold δ]
Feature-Squeeze — FAR=0.486, DR(FGSM)=0.979
Autoencoder     — FAR=0.045, DR(FGSM)=1.000













# logs after adding the fs_threshold code:



# train_autoencoder.py:
[INFO] Using device: cuda
[INFO] Loading data...
[INFO] Dataset sizes — train: 54000, val: 6000, test: 10000
[INFO] Autoencoder initialized:
AutoencoderDetector(
  (encoder): Sequential(
    (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(16, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (3): ReLU()
  )
  (decoder): Sequential(
    (0): ConvTranspose2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (1): ReLU()
    (2): ConvTranspose2d(16, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (3): Tanh()
  )
)
[INFO] Starting training for 10 epochs...
  [Epoch 1] Batch 100/844 — Batch MSE: 0.016636
  [Epoch 1] Batch 200/844 — Batch MSE: 0.006041
  [Epoch 1] Batch 300/844 — Batch MSE: 0.004226
  [Epoch 1] Batch 400/844 — Batch MSE: 0.003184
  [Epoch 1] Batch 500/844 — Batch MSE: 0.002448
  [Epoch 1] Batch 600/844 — Batch MSE: 0.002058
  [Epoch 1] Batch 700/844 — Batch MSE: 0.001552
  [Epoch 1] Batch 800/844 — Batch MSE: 0.001346
  [Epoch 1] Batch 844/844 — Batch MSE: 0.001248
[RESULT] Epoch 1 completed in 64.7s — Train MSE: 0.032199 — Val MSE: 0.001221
  [Epoch 2] Batch 100/844 — Batch MSE: 0.001021
  [Epoch 2] Batch 200/844 — Batch MSE: 0.000813
  [Epoch 2] Batch 300/844 — Batch MSE: 0.000749
  [Epoch 2] Batch 400/844 — Batch MSE: 0.000595
  [Epoch 2] Batch 500/844 — Batch MSE: 0.000550
  [Epoch 2] Batch 600/844 — Batch MSE: 0.000503
  [Epoch 2] Batch 700/844 — Batch MSE: 0.000452
  [Epoch 2] Batch 800/844 — Batch MSE: 0.000461
  [Epoch 2] Batch 844/844 — Batch MSE: 0.000447
[RESULT] Epoch 2 completed in 76.5s — Train MSE: 0.000678 — Val MSE: 0.000427
  [Epoch 3] Batch 100/844 — Batch MSE: 0.000415
  [Epoch 3] Batch 200/844 — Batch MSE: 0.000366
  [Epoch 3] Batch 300/844 — Batch MSE: 0.000360
  [Epoch 3] Batch 400/844 — Batch MSE: 0.000328
  [Epoch 3] Batch 500/844 — Batch MSE: 0.000318
  [Epoch 3] Batch 600/844 — Batch MSE: 0.000319
  [Epoch 3] Batch 700/844 — Batch MSE: 0.000311
  [Epoch 3] Batch 800/844 — Batch MSE: 0.000284
  [Epoch 3] Batch 844/844 — Batch MSE: 0.000278
[RESULT] Epoch 3 completed in 74.2s — Train MSE: 0.000339 — Val MSE: 0.000274
  [Epoch 4] Batch 100/844 — Batch MSE: 0.000255
  [Epoch 4] Batch 200/844 — Batch MSE: 0.000252
  [Epoch 4] Batch 300/844 — Batch MSE: 0.000237
  [Epoch 4] Batch 400/844 — Batch MSE: 0.000219
  [Epoch 4] Batch 500/844 — Batch MSE: 0.000185
  [Epoch 4] Batch 600/844 — Batch MSE: 0.000186
  [Epoch 4] Batch 700/844 — Batch MSE: 0.000171
  [Epoch 4] Batch 800/844 — Batch MSE: 0.000158
  [Epoch 4] Batch 844/844 — Batch MSE: 0.000149
[RESULT] Epoch 4 completed in 78.8s — Train MSE: 0.000210 — Val MSE: 0.000158
  [Epoch 5] Batch 100/844 — Batch MSE: 0.000153
  [Epoch 5] Batch 200/844 — Batch MSE: 0.000142
  [Epoch 5] Batch 300/844 — Batch MSE: 0.000134
  [Epoch 5] Batch 400/844 — Batch MSE: 0.000129
  [Epoch 5] Batch 500/844 — Batch MSE: 0.000136
  [Epoch 5] Batch 600/844 — Batch MSE: 0.000126
  [Epoch 5] Batch 700/844 — Batch MSE: 0.000120
  [Epoch 5] Batch 800/844 — Batch MSE: 0.000116
  [Epoch 5] Batch 844/844 — Batch MSE: 0.000116
[RESULT] Epoch 5 completed in 76.9s — Train MSE: 0.000134 — Val MSE: 0.000116
  [Epoch 6] Batch 100/844 — Batch MSE: 0.000115
  [Epoch 6] Batch 200/844 — Batch MSE: 0.000106
  [Epoch 6] Batch 300/844 — Batch MSE: 0.000105
  [Epoch 6] Batch 400/844 — Batch MSE: 0.000102
  [Epoch 6] Batch 500/844 — Batch MSE: 0.000105
  [Epoch 6] Batch 600/844 — Batch MSE: 0.000101
  [Epoch 6] Batch 700/844 — Batch MSE: 0.000095
  [Epoch 6] Batch 800/844 — Batch MSE: 0.000091
  [Epoch 6] Batch 844/844 — Batch MSE: 0.000088
[RESULT] Epoch 6 completed in 80.1s — Train MSE: 0.000105 — Val MSE: 0.000094
  [Epoch 7] Batch 100/844 — Batch MSE: 0.000089
  [Epoch 7] Batch 200/844 — Batch MSE: 0.000086
  [Epoch 7] Batch 300/844 — Batch MSE: 0.000093
  [Epoch 7] Batch 400/844 — Batch MSE: 0.000087
  [Epoch 7] Batch 500/844 — Batch MSE: 0.000087
  [Epoch 7] Batch 600/844 — Batch MSE: 0.000085
  [Epoch 7] Batch 700/844 — Batch MSE: 0.000082
  [Epoch 7] Batch 800/844 — Batch MSE: 0.000085
  [Epoch 7] Batch 844/844 — Batch MSE: 0.000075
[RESULT] Epoch 7 completed in 81.0s — Train MSE: 0.000086 — Val MSE: 0.000078
  [Epoch 8] Batch 100/844 — Batch MSE: 0.000075
  [Epoch 8] Batch 200/844 — Batch MSE: 0.000075
  [Epoch 8] Batch 300/844 — Batch MSE: 0.000072
  [Epoch 8] Batch 400/844 — Batch MSE: 0.000075
  [Epoch 8] Batch 500/844 — Batch MSE: 0.000071
  [Epoch 8] Batch 600/844 — Batch MSE: 0.000068
  [Epoch 8] Batch 700/844 — Batch MSE: 0.000075
  [Epoch 8] Batch 800/844 — Batch MSE: 0.000070
  [Epoch 8] Batch 844/844 — Batch MSE: 0.000069
[RESULT] Epoch 8 completed in 79.5s — Train MSE: 0.000073 — Val MSE: 0.000067
  [Epoch 9] Batch 100/844 — Batch MSE: 0.000066
  [Epoch 9] Batch 200/844 — Batch MSE: 0.000062
  [Epoch 9] Batch 300/844 — Batch MSE: 0.000061
  [Epoch 9] Batch 400/844 — Batch MSE: 0.000063
  [Epoch 9] Batch 500/844 — Batch MSE: 0.000060
  [Epoch 9] Batch 600/844 — Batch MSE: 0.000060
  [Epoch 9] Batch 700/844 — Batch MSE: 0.000059
  [Epoch 9] Batch 800/844 — Batch MSE: 0.000058
  [Epoch 9] Batch 844/844 — Batch MSE: 0.000062
[RESULT] Epoch 9 completed in 77.2s — Train MSE: 0.000062 — Val MSE: 0.000059
  [Epoch 10] Batch 100/844 — Batch MSE: 0.000053
  [Epoch 10] Batch 200/844 — Batch MSE: 0.000057
  [Epoch 10] Batch 300/844 — Batch MSE: 0.000055
  [Epoch 10] Batch 400/844 — Batch MSE: 0.000052
  [Epoch 10] Batch 500/844 — Batch MSE: 0.000051
  [Epoch 10] Batch 600/844 — Batch MSE: 0.000052
  [Epoch 10] Batch 700/844 — Batch MSE: 0.000051
  [Epoch 10] Batch 800/844 — Batch MSE: 0.000049
  [Epoch 10] Batch 844/844 — Batch MSE: 0.000048
[RESULT] Epoch 10 completed in 78.8s — Train MSE: 0.000054 — Val MSE: 0.000051
[INFO] Computed 95th-percentile recon-error threshold: 0.000072
[INFO] Saved weights to checkpoints/autoencoder.pth
[INFO] Saved threshold to checkpoints/threshold.json
[DONE] Autoencoder training and threshold calibration complete.
C:\Users\Abhis\.conda\envs\torch_gpu_env\lib\site-packages\transformers\models\vit\modeling_vit.py:277: UserWarning: 1Torch was not compiled with flash attention. (Triggered internally at C:\cb\pytorch_1000000000000\work\aten\src\ATen\native\transformers\cuda\sdp_utils.cpp:555.)
  context_layer = torch.nn.functional.scaled_dot_product_attention(
[INFO] FS threshold (95th pct val): 5.807629e-04
[INFO] Saved FS threshold to checkpoints/fs_threshold.json





# make_attacks.py:
[INFO] Generating adversarial examples on cuda
[INFO] Test set size: 10000 samples
C:\Users\Abhis\.conda\envs\torch_gpu_env\lib\site-packages\transformers\models\vit\modeling_vit.py:277: UserWarning: 1Torch was not compiled with flash attention. (Triggered internally at C:\cb\pytorch_1000000000000\work\aten\src\ATen\native\transformers\cuda\sdp_utils.cpp:555.)
  context_layer = torch.nn.functional.scaled_dot_product_attention(
[INFO] Crafting FGSM examples...
FGSM: 100%|███████████████████████████████████████████████████| 625/625 [04:35<00:00,  2.27it/s]
[INFO] Crafting PGD examples...
PGD: 100%|████████████████████████████████████████████████████| 625/625 [24:08<00:00,  2.32s/it]
[DONE] Adversarial datasets saved in checkpoints/



# eval_detectors.py
[INFO] Evaluating on cuda
.\eval_detectors.py:77: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  ae.load_state_dict(torch.load("checkpoints/autoencoder.pth", map_location=device))
[INFO] AE δ threshold: 7.230532e-05
[INFO] FS τ threshold: 5.807629e-04
.\eval_detectors.py:101: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  X_fgsm, Y_fgsm = torch.load("checkpoints/adv_fgsm.pt")
.\eval_detectors.py:102: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  X_pgd,  Y_pgd  = torch.load("checkpoints/adv_pgd.pt")
[INFO] Loaded adv sets: FGSM(10000), PGD(10000)
Scoring FeatureSqueezeDetector:   0%|                                    | 0/40 [00:00<?, ?it/s]C:\Users\Abhis\.conda\envs\torch_gpu_env\lib\site-packages\transformers\models\vit\modeling_vit.py:277: UserWarning: 1Torch was not compiled with flash attention. (Triggered internally at C:\cb\pytorch_1000000000000\work\aten\src\ATen\native\transformers\cuda\sdp_utils.cpp:555.)
  context_layer = torch.nn.functional.scaled_dot_product_attention(
Scoring FeatureSqueezeDetector: 100%|███████████████████████████| 40/40 [03:13<00:00,  4.83s/it]
Scoring FeatureSqueezeDetector: 100%|███████████████████████████| 40/40 [02:30<00:00,  3.77s/it]
Scoring FeatureSqueezeDetector: 100%|███████████████████████████| 40/40 [02:31<00:00,  3.80s/it]
Scoring AutoencoderDetector: 100%|██████████████████████████████| 40/40 [01:02<00:00,  1.57s/it]
Scoring AutoencoderDetector: 100%|██████████████████████████████| 40/40 [00:08<00:00,  4.50it/s]
Scoring AutoencoderDetector: 100%|██████████████████████████████| 40/40 [00:09<00:00,  4.38it/s]
✅ Saved ROC & PR plots and metrics in results/