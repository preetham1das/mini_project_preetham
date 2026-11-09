# Kaggle Amazon Deforestation Experiment Report

**Status: NOT TRAINED.** This file is a report template; no sample counts, metrics, or model results are fabricated.

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

Fill this section from the notebook’s generated `dataset_report.json` after download and inspection.

- Downloaded archive size/hash: pending
- Extracted TIFF count / total extracted bytes: pending
- Cloud-free Sentinel-1 / Sentinel-2 / mask counts: pending
- Cloudy Sentinel-1 / Sentinel-2 / mask counts: pending
- Verified paired tile count: pending
- Actual input band counts and value ranges: pending
- Mask unique values, positive/background mapping, and counts: pending
- CRS, affine-transform, dimensions, and alignment checks: pending
- Geographic train / validation / test block and patch counts: pending
- Visual inspection status: pending

## Training and evaluation

- GPU model / VRAM: pending
- Batch size / mixed precision: pending
- Epochs completed / early stopping: pending
- Best checkpoint: pending
- Test IoU: pending
- Test Dice/F1: pending
- Test precision / recall / accuracy: pending
- Test confusion matrix (TN, FP, FN, TP): pending
- Training history and plots: pending
- Limitations / failed samples: pending

The notebook writes measured values and artifact locations to the Drive experiment output directory. Copy the generated report here after the actual run if you want this repository file to contain final results.
