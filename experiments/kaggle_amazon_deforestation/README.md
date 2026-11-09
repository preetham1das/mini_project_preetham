# Kaggle Amazon Deforestation: Colab experiment

This is a separate PyTorch experiment using only the Kaggle dataset `akhilchibber/deforestation-detection-dataset`. It does not modify or train the existing Stage 1 / Stage 2 project models and it never uses the 334 Forest-Change pairs.

## Research implementation inspected

The dataset author’s public research repository includes separate cloud-free Sentinel-1, Sentinel-2, and late-fusion notebooks. The late-fusion model uses distinct optical and radar encoders and decoders, then concatenates decoded features for pixel classification. The notebook declares 4 Sentinel-2 bands, 2 Sentinel-1 bands, 256×256 patches, and a two-class output. This experiment preserves that modality-specific, late-fusion structure in PyTorch. It does **not** treat Sentinel-1 and Sentinel-2 as before/after dates. The original notebooks randomly split patches from selected training grids for validation and hard-code two grid IDs for test; this version instead assigns complete geospatial raster blocks to train/validation/test before creating patches.

Source material: [author’s research notebooks](https://github.com/akhilchibber/Deforestation-Detection) and [M.Sc. thesis](https://essay.utwente.nl/91435/1/CHHIBBER_ITC.pdf). The original notebooks are TensorFlow/Keras code; the local `model.py` is a new PyTorch implementation, not copied weights or a pretrained model.

## Files

- `kaggle_amazon_colab.ipynb`: self-contained Colab workflow for authentication, Drive storage, Kaggle download, archive verification, dataset inspection, alignment, spatial splits, previews, training, evaluation and reports.
- `model.py`: two-stream late-fusion SegNet with separate Sentinel-1 and Sentinel-2 encoders/decoders.
- `dataset.py`: GeoTIFF metadata/alignment checks, spatial grouping, patch indexing, train-only robust scaling and masked PyTorch loading.
- `EXPERIMENT_REPORT.md`: status scaffold. It contains no fabricated training results; the notebook writes measured results after a real Colab run.
- `research_metrics.json`: recorded V2 metrics supplied from the Colab report, shown separately from each inference result. It includes an audit note about the unresolved training patch-count discrepancy.

## Use the trained model in ForestWatch

The existing app exposes a separate **S1 + S2 Experiment** page at `#/kaggle-amazon` and the `/api/predict/kaggle-amazon` inference endpoint. Keep `late_fusion_segnet_v2_corrected_best.pt` and its matching `normalization.json` together under `checkpoints/kaggle_amazon/`. Upload two co-registered GeoTIFFs with the same CRS, affine transform, and dimensions: 2 Sentinel-1 bands and 4 Sentinel-2 bands. The endpoint returns preview images and metrics, plus a downloadable single-band GeoTIFF using the input grid/CRS and `255` for NoData. The research metrics are reported separately and are never calculated from uploaded images.

### Live Sentinel-1 retrieval and calibration status

The experiment page can retrieve Sentinel-1 GRD VV/VH from the Copernicus Data Space Process API. Upload the four-band Sentinel-2 GeoTIFF first; its footprint, CRS, dimensions, and affine transform define the requested AOI and output grid. Choose a date interval of up to 90 days. The backend reuses the application's existing OAuth helper, searches the Sentinel-1 Catalog, requests dual polarization (`DV`), and returns a float32 GeoTIFF with bands ordered `VV`, `VH` and NoData `-9999`.

The retrieved product explicitly uses `GAMMA0_ELLIPSOID` in `LINEAR_POWER` units. **It is intentionally blocked from V2 inference for now.** The Colab V2 loader reads the Kaggle Sentinel-1 rasters directly as stored and applies the saved train-only percentiles, but neither the notebook nor saved experiment artifacts record the source coefficient/units. The reported V2 ranges also include negative values, which cannot be the Sentinel Hub linear-power values. Do not remove this guard or treat gamma0 linear power as compatible until the original Kaggle raster metadata/calibration has been verified and the preprocessing is reproduced.

## Colab setup and run

1. Open `kaggle_amazon_colab.ipynb` in Google Colab. Select **Runtime → Change runtime type → T4 GPU** (or any available free GPU). The notebook checks the assigned device and VRAM; no paid runtime is enabled or required by the code.
2. In Colab’s **Secrets** panel, create `KAGGLE_API_TOKEN` using a Kaggle API token from your Kaggle account settings. Enable notebook access to that secret. Never paste the token into a code cell or commit it.
3. Run cells from top to bottom. Approve Google Drive mount when prompted. The archive and extracted files are stored under `/content/drive/MyDrive/Deforestation_Research/Kaggle_Amazon/`.
4. The first download is approximately 12.37 GB, before extraction. Check that your Drive has space for both the original archive and extracted files. The archive is retained; extraction is resumable/idempotent after the verification manifest is written.
5. Review the actual inventory, class values, CRS/transform checks, spatial split map, and sample visualizations before running the training cell. The workflow stops if it cannot verify cloud-free Sentinel-1/Sentinel-2/mask pairs, class semantics, or three geographic splits.
6. Training saves the best checkpoint, an epoch-resumable checkpoint, metrics/history, test predictions/plots, and a measured `EXPERIMENT_REPORT.md` to the experiment output folder in Drive. If Colab disconnects, reconnect, mount Drive and run cells again; training resumes from the last completed epoch checkpoint.

The data folder contains only the Kaggle dataset and derived files. Cloudy imagery is counted and inspected; training uses the author-provided cloud-free domain unless you deliberately change the documented configuration. The masks are read as the source PRODES labels; pixels outside the verified `{0,1}` class mapping or invalid in either image are excluded from loss and metrics.

## Training design

- Separate SAR and optical streams; model inputs are not temporal pairs.
- Original research band counts are checked (expected 2 Sentinel-1, 4 Sentinel-2); actual file band counts control model construction after verification.
- 256×256 windows; non-overlapping stride for the primary split to avoid duplicated neighboring patches.
- Spatial raster blocks are assigned to splits before patching. Realized split ratios are reported; a small number of blocks may make them differ from 70/15/15.
- Per-band 1st/99th percentile scaling is fitted using train windows only. Values are clipped to `[0,1]`; paired invalid pixels are ignored.
- BCE-with-logits plus Dice loss, modest positive weighting, train-only flips/90° rotations, validation-based early stopping, best and resume checkpoints.
- Test metrics use a fixed 0.5 probability threshold and are computed once after training. No result is included before you run the experiment.

## Important limitations

The Kaggle page and raster metadata must be checked during the notebook run. If folder names, dimensions, CRS, class values, or mask alignment differ from the author’s research notebooks, the notebook records the discrepancy and stops instead of silently pairing, resampling, or relabeling. Cloud-free inputs do not establish performance under cloudy conditions; the cloudy domain is inspected separately and should be treated as an additional robustness experiment. PRODES annual masks and satellite grids can still contain source-label uncertainty even when raster alignment is exact.
