# Kaggle Amazon Deforestation Experiment Report

**Status: Trained in Google Colab.** No training was run on this project machine. The V2 values below were transcribed from the user-provided Colab report generated on 2026-11-08; they have not been independently reproduced here.

## Experiment definition

- Dataset: Kaggle `akhilchibber/deforestation-detection-dataset`
- Data location: `/content/drive/MyDrive/Deforestation_Research/Kaggle_Amazon/`
- Model: separate Sentinel-1/Sentinel-2 late-fusion SegNet
- Sentinel-1: independent SAR input branch
- Sentinel-2: independent optical input branch
- Fusion: decoded modality features concatenated before binary output
- Training environment: Google Colab only; GPU/runtime to be recorded from the notebook
- Existing Forest-Change dataset: not used

## Dataset verification

- Kaggle source: `akhilchibber/deforestation-detection-dataset`.
- Archive size/hash reported: 10,844,915,792 bytes; SHA-256 `dcbb105e09e957950ca83611f42bbd0859d6f2140569b2958ad9ca581333a10d`.
- Extracted files/bytes: 96 / 12,370,935,899; TIFF files: 85.
- Cloud-free aligned S1/S2/mask pairs: 17; cloudy pairs inspected but excluded: 17.
- Bands: Sentinel-1=2, Sentinel-2=4; alignment failures reported: 0.
- Mask counts reported: value 1=98,500,982; value 2=155,254,404; value 0=6 (NoData). Mapping: 1 positive, 2 negative, 0 ignored.
- Geographic tiles: train=9, val=4, test=4. Reported patches: train=2,904, val=484, test=484; the training patch count is under audit as described below.

## Training and evaluation

- GPU model / VRAM: Tesla T4 / 14.6 GiB (reported).
- Batch size / mixed precision: 2 / enabled (reported).
- Epochs completed: 13; best checkpoint epoch: 5 (reported).
- Checkpoint path in Colab Drive: `models/late_fusion_segnet_v2_corrected_best.pt`.
- Held-out test metrics and confusion matrix: see the recorded V2 section below.
- Limitation: these values are transcribed from the user's Colab report and not independently reproduced from the Drive artifacts in this workspace.

## Recorded V2 run (user-provided Colab report)

- Run: `late_fusion_segnet_v2_corrected`, generated `2026-11-08T13:52:08.214449+00:00`.
- Ground-truth mapping reported: raw `0` = NoData/ignored; `1` = deforestation; `2` = non-deforestation.
- Best validation checkpoint: epoch 5; IoU 0.860067, Dice/F1 0.924770, precision 0.924421, recall 0.925119, accuracy 0.930719.
- Geographic test at threshold 0.5: IoU 0.848883, Dice/F1 0.918266, precision 0.921625, recall 0.914932, accuracy 0.931066.
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[17250270, 1044521], [1142012, 12282620]]`.
- Reported run: Tesla T4, batch size 2, mixed precision, 13 completed epochs.
- Audit limitation: reported 2,904 training patches across 9 tiles do not reconcile with 2,816×2,816 tiles and non-overlapping 256×256 patches. Verify the saved patch index and split provenance before treating these scores as final claims.

These values are an attributed record of the supplied Colab output, not an independent validation. Per-image inference in the app is displayed separately from these research metrics.
