# Project Diary — 10 Weeks

**Project Title:** AI-Based Forest Cover Segmentation and Temporal Deforestation Detection Using Deep Convolutional Networks with Live Copernicus Sentinel-2 GIS Analysis

**Student:** ___________________________
**Roll No. / USN:** ___________________________
**Guide:** ___________________________
**Department:** ___________________________
**Academic Year:** 2026–27

**Diary Period:** Week 1 (03 Aug 2026) – Week 10 (11 Oct 2026)

---

## How to use this document

Each week follows the same structure so it can be copied directly into a printed diary:

- **Objectives** — what the week set out to achieve
- **Work Done** — the concrete tasks, files and features
- **Tools & Technologies** — what was actually used
- **Outcome / Deliverable** — the artefact produced
- **Challenges & Resolution** — genuine problems hit and how they were solved
- **Learning Outcome** — the takeaway to write in the reflection column
- **Approx. Hours** — adjust to your own record
- **Next Week Plan** — continuity between entries

**Weeks 1–5** cover groundwork — literature study, dataset analysis, prototyping and architecture design. **Weeks 6–10** map directly onto the repository's commit history (first commit 09 Sep 2026). Work completed after the diary period is summarised in the appendix at the end, in case your submission date extends past 11 October.

---

## Summary Table

| Week | Date Range | Focus | Key Deliverable |
|:---:|---|---|---|
| 1 | 03–09 Aug | Problem definition, literature survey, tooling | Problem statement & environment |
| 2 | 10–16 Aug | Dataset acquisition and exploratory analysis | Dataset audit notes |
| 3 | 17–23 Aug | U-Net study and first prototype | Working segmentation prototype |
| 4 | 24–30 Aug | Metrics, losses and baseline evaluation | Evaluation harness |
| 5 | 31 Aug–06 Sep | Two-stage architecture design | Design document |
| 6 | 07–13 Sep | Repository setup, Stage 1 training, first UI | Trained Stage 1 model + web demo |
| 7 | 14–20 Sep | Stage 2 temporal model, imbalance audit | 9-channel Temporal U-Net |
| 8 | 21–27 Sep | Uploads, Copernicus API, AOI workflow | Live Sentinel-2 search & timeline |
| 9 | 28 Sep–04 Oct | SwinIR SR, SCL cloud analytics, docs | Super-resolution + cloud stats |
| 10 | 05–11 Oct | Consolidation, evaluation, milestone demo | Measured metrics + working system |

---

## Week 1 — Problem Definition, Literature Survey and Tooling
**Date Range:** 03 Aug 2026 – 09 Aug 2026

**Objectives**
1. Convert the broad topic of "deforestation detection" into a precise, achievable problem statement.
2. Survey existing approaches and identify why they fail in practice.
3. Set up the development environment.

**Work Done**
- Reviewed literature on forest cover mapping and change detection: semantic segmentation with U-Net and its variants, post-classification comparison (PCC), and bi-temporal deep change-detection networks.
- Identified the two recurring failure modes reported in the literature: severe class imbalance, since deforestation typically affects under 5% of a scene, and error multiplication in PCC, where two independently segmented masks are subtracted and their errors compound.
- Wrote the initial problem statement: segment baseline forest canopy from a single image, then detect forest loss between two acquisition dates.
- Installed the Python 3.13 toolchain, PyTorch (CPU build), torchvision, NumPy, Pillow, OpenCV, Matplotlib, Flask and Git, and verified that a trivial forward pass ran.
- Registered for a free Copernicus Data Space Ecosystem account to evaluate live Sentinel-2 data access, and reviewed the STAC and Process API documentation.

**Tools & Technologies:** Python 3.13, PyTorch, NumPy, OpenCV, Pillow, Flask, Git, VS Code, Copernicus Data Space documentation.

**Outcome / Deliverable:** An agreed problem statement, a literature summary, and a working development environment.

**Challenges & Resolution**
- *Scope creep:* the initial idea covered land-cover classification, fire detection and multi-date time series. Narrowed the scope to two well-defined stages so the project could be completed and evaluated rigorously within the available time.
- *Hardware constraint:* no dedicated GPU was available, so CPU-only training was assumed from the start. This influenced later decisions on image size (256×256) and model capacity.

**Learning Outcome:** Learned that a narrow, measurable problem statement is far more valuable than a broad one, and that hardware constraints should shape architecture choices from day one rather than being discovered later.

**Approx. Hours:** 14

**Next Week Plan:** Obtain and thoroughly understand the datasets before writing any training code.

---

## Week 2 — Dataset Acquisition and Exploratory Analysis
**Date Range:** 10 Aug 2026 – 16 Aug 2026

**Objectives**
1. Acquire suitable datasets for both stages.
2. Understand the label encoding, spatial resolution and class distribution.
3. Decide on train/validation/test splits.

**Work Done**
- Obtained the **Amazon Forest Dataset** for Stage 1: 30 training images with masks, 15 validation images with masks and 15 unlabelled test TIFFs.
- Obtained the **Forest-Change dataset** for Stage 2: 334 bi-temporal RGB pairs at 480×480, already split into 270 training, 31 validation and 33 test pairs.
- Inspected the TIFF files to determine band count, bit depth and mask encoding. Discovered that the Stage 1 masks use raw value **2 for forest** and **1 for background**, not the conventional 0/1 encoding.
- Wrote small exploratory scripts to display sample images beside their masks and confirm that the remapping was correct.
- Computed the class distribution of the Stage 2 change masks and found that only about **4.8%** of pixels were labelled as deforestation, an imbalance ratio of roughly **19.9 : 1**.
- Recorded the split sizes, resolutions and label conventions in a dataset notes file for later reference.

**Tools & Technologies:** Python, NumPy, Pillow, tifffile, Matplotlib, OpenCV.

**Outcome / Deliverable:** Documented dataset characteristics and a verified label-remapping rule.

**Challenges & Resolution**
- *Inverted masks:* the first visualisation showed forest and background swapped because the mask values were 1 and 2 rather than 0 and 1. Recorded the correct mapping explicitly (2 → forest, 1 → background) so it could be applied consistently in every loader.
- *Unlabelled test data:* the Stage 1 test split had no ground-truth masks, which meant quantitative evaluation would have to rely on the 15-image validation split. Accepted this limitation and noted it for the report.

**Learning Outcome:** Learned that dataset inspection must precede model design — label encoding, resolution and class balance determine everything downstream, and a wrong assumption here silently invalidates weeks of training.

**Approx. Hours:** 14

**Next Week Plan:** Implement the U-Net architecture and a dataset loader, and get a first training run working end to end.

---

## Week 3 — U-Net Study and First Working Prototype
**Date Range:** 17 Aug 2026 – 23 Aug 2026

**Objectives**
1. Implement a U-Net from scratch to understand every component.
2. Build a PyTorch dataset and dataloader for the forest data.
3. Achieve a first end-to-end training run.

**Work Done**
- Implemented the U-Net architecture with four downsampling stages (32 → 512 channels), four transposed-convolution upsampling stages, and skip connections that concatenate encoder features into the decoder.
- Wrote a dataset class that loads image–mask pairs, applies the confirmed label remapping, resizes to 256×256 and normalises pixel values to [0, 1].
- Built the training loop: forward pass, BCE loss, backpropagation, validation pass, and checkpoint saving.
- Ran an initial training experiment on the 30 training images to confirm the pipeline worked. Loss decreased steadily and masks began resembling tree cover within a few epochs.
- Inspected predictions qualitatively and found the model was reasonably good at large canopy areas but weak on boundaries.

**Tools & Technologies:** PyTorch, torchvision, NumPy, Matplotlib, Pillow.

**Outcome / Deliverable:** A functioning segmentation prototype with reproducible training loop and saved checkpoints.

**Challenges & Resolution**
- *Shape mismatch in the decoder:* concatenating encoder and decoder features failed until the channel dimensions and spatial sizes were matched exactly. Resolved by carefully ordering the transposed convolution before the concatenation and doubling the expected input channels of each decoder block.
- *Unstable early training:* plain BCE produced predictions that flickered between all-forest and all-background on small batches. Addressed temporarily by lowering the learning rate, which motivated the more principled loss work in the following week.

**Learning Outcome:** Learned the U-Net skip-connection mechanics properly by implementing them rather than importing them, and understood why decoder channel arithmetic is the most common source of shape errors.

**Approx. Hours:** 16

**Next Week Plan:** Replace the ad-hoc loss with a principled objective and build a proper evaluation harness.

---

## Week 4 — Metrics, Loss Functions and Baseline Evaluation
**Date Range:** 24 Aug 2026 – 30 Aug 2026

**Objectives**
1. Implement segmentation metrics correctly.
2. Improve the loss function for imbalanced binary masks.
3. Establish a baseline that later work can be measured against.

**Work Done**
- Implemented IoU (Jaccard index), Dice coefficient, precision, recall and pixel accuracy from the confusion matrix, and validated them against hand-computed values on toy masks.
- Compared objectives: plain binary cross-entropy, BCE with a positive class weight, and a combined BCE + soft Dice loss.
- Observed that pixel accuracy was a misleading metric — a model predicting background everywhere still scored above 90% — and established IoU and Dice as the primary measures.
- Ran the Stage 1 baseline to convergence, monitoring validation IoU per epoch, and saved the best checkpoint.
- Noted that the best validation result appeared very early in training, which suggested the 30-image training set was too small to support long training.

**Tools & Technologies:** PyTorch, NumPy, Matplotlib.

**Outcome / Deliverable:** A reusable metrics module and a documented Stage 1 baseline.

**Challenges & Resolution**
- *Metric implementation errors:* the first Dice implementation averaged per-image scores in a way that disagreed with the global score. Fixed by computing global true positives, false positives and false negatives across the whole split, and reporting both per-image and global values.
- *Early overfitting:* validation IoU peaked and then declined. Chose to save the best checkpoint by validation IoU rather than by final epoch, and recorded the early peak as a genuine limitation of the dataset size.

**Learning Outcome:** Learned that metric choice changes conclusions — accuracy hides failure on rare classes — and that checkpoint selection must be driven by validation performance, not by training length.

**Approx. Hours:** 16

**Next Week Plan:** Design the two-stage architecture and decide how Stage 2 should model temporal change.

---

## Week 5 — Two-Stage Architecture Design
**Date Range:** 31 Aug 2026 – 06 Sep 2026

**Objectives**
1. Decide how to detect deforestation between two dates.
2. Compare post-classification comparison against direct temporal modelling.
3. Freeze the system design before implementation.

**Work Done**
- Prototyped post-classification comparison: segment T1 and T2 independently with the Stage 1 model, then subtract the masks. Results were very poor — small per-image errors produced large spurious change regions, confirming the error-multiplication problem identified in the literature.
- Designed an alternative: a single network that consumes both dates simultaneously. Defined the input as `[T1 RGB, T2 RGB, |T1 − T2|]`, giving nine channels, so the first convolution receives an explicit spectral-difference signal.
- Specified the Stage 2 model as the same U-Net topology with a 9-channel input, keeping the parameter count comparable to Stage 1 so the two stages remain consistent.
- Planned the system architecture: a Flask backend serving both models over REST, a React frontend with separate workbenches for each stage, and later a GIS view for live satellite data.
- Defined the evaluation protocol — fixed threshold of 0.5, fixed random seed of 42, model selection on validation, final reporting on the held-out test split.
- Wrote the design document and prepared the repository structure for implementation.

**Tools & Technologies:** PyTorch, Markdown, Git, Mermaid (for architecture diagrams).

**Outcome / Deliverable:** A frozen architecture design and evaluation protocol.

**Challenges & Resolution**
- *PCC failure was severe:* subtracting two masks produced change maps dominated by noise rather than real forest loss. Rather than tuning PCC, decided to abandon it as the primary method and keep it only as a documented baseline for comparison.
- *Where to put the difference channel:* considered computing the difference inside the network versus supplying it as input. Chose to supply it explicitly, since it gives the first layer a direct inductive bias and keeps the model simple to reason about.

**Learning Outcome:** Learned that architectural decisions should be justified by a measured failure, not by preference — the PCC experiment is what made the 9-channel design defensible rather than arbitrary.

**Approx. Hours:** 14

**Next Week Plan:** Create the repository, train the final Stage 1 model, and build the first web interface.

---

## Week 6 — Repository Setup, Stage 1 Training and First Interface
**Date Range:** 07 Sep 2026 – 13 Sep 2026

**Objectives**
1. Create the project repository with a clean folder structure.
2. Train and save the final Stage 1 checkpoint.
3. Deliver a first working web interface.

**Work Done**
- Initialised the Git repository and created the folder structure: `src/` for the model pipeline, `data/` for datasets, `models/` for checkpoints, `outputs/` for evaluation artefacts, and `templates/` and `static/` for the interface.
- Configured `.gitignore` for Python caches, the virtual environment and local environment files.
- Finalised `src/dataset.py`, `src/model.py` and `src/train.py`, and trained Stage 1 to completion. The best checkpoint, saved at epoch 5, was written to `models/best_unet_model.pth` (7,763,041 parameters).
- Built a Flask application exposing a prediction endpoint that accepts an uploaded image, returns a forest mask and reports the forest pixel percentage.
- Created the first HTML/CSS interface with an upload control and a side-by-side result view, and applied a white-and-emerald visual theme.
- Committed the working baseline and a full project backup to version control.

**Tools & Technologies:** Git, GitHub, Flask, Jinja2, HTML, CSS, PyTorch.

**Outcome / Deliverable:** A trained Stage 1 model and a functioning browser-based segmentation demo.

**Challenges & Resolution**
- *Encoding errors in the browser upload path:* non-image or mislabelled files caused unhandled exceptions and returned HTML error pages instead of JSON. Added explicit validation and JSON error responses so the frontend could display a meaningful message.
- *Slow CPU inference:* each request took noticeably longer than expected. Reduced the working resolution to 256×256 and avoided reloading the model per request by loading it once at startup.

**Learning Outcome:** Learned how much engineering sits between a trained checkpoint and a usable application — model loading, input validation, error contracts and response encoding all had to be handled explicitly.

**Approx. Hours:** 18

**Next Week Plan:** Implement the Stage 2 temporal model based on the frozen design and begin the multi-view dashboard.

---

## Week 7 — Stage 2 Temporal Model and Class Imbalance Handling
**Date Range:** 14 Sep 2026 – 20 Sep 2026

**Objectives**
1. Implement and train the 9-channel Temporal U-Net.
2. Handle the measured 19.9 : 1 class imbalance correctly.
3. Expand the interface into a multi-stage dashboard.

**Work Done**
- Wrote `src/temporal_dataset.py` supporting both a 6-channel mode (`[T1, T2]`) and a 9-channel mode (`[T1, T2, |T1 − T2|]`), so the two input strategies could be compared fairly.
- Implemented `TemporalUNet` in `src/temporal_model.py` with configurable input channels and multiple objectives: weighted BCE + Dice, focal loss, and focal + Dice combined.
- Applied the positive class weight of **19.94** derived from the Week 2 imbalance audit.
- Wrote `src/train_temporal_v2.py` to run controlled experiments sequentially — 6-channel versus 9-channel inputs, and weighted BCE versus focal loss — each saving its own history, curves and checkpoint, with the best validation IoU promoted to `models/best_temporal_model.pth`.
- Trained the 9-channel focal + Dice configuration; the best checkpoint was saved at epoch 21 with a validation IoU of 0.5043.
- Updated the backend to detect the checkpoint's input channel count and build the tensor accordingly.
- Reworked the frontend into a dashboard with dedicated Stage 1 and Stage 2 workbenches, and added an initial GIS control panel and map viewer.

**Tools & Technologies:** PyTorch, NumPy, pandas, Matplotlib, Flask, FastAPI, React, GitHub pull requests.

**Outcome / Deliverable:** A trained 9-channel Temporal U-Net (7,764,769 parameters) and a documented comparison of input and loss strategies.

**Challenges & Resolution**
- *All-zero predictions:* the first Stage 2 run produced empty masks because background dominated the loss. Combining the positive weight with focal loss and a soft Dice term restored meaningful gradients on the rare change class.
- *Checkpoint key mismatch:* earlier experimental runs named the upsampling blocks `cu1`–`cu4` while the final architecture used `conv_up1`–`conv_up4`. Added a key-remapping step in the loader so older checkpoints load without retraining.
- *Diverged branches:* parallel work on the theme and the GIS panel created conflicts. Began resolving them through pull requests rather than direct commits to `main`.

**Learning Outcome:** Learned that a model predicting "no change" everywhere can exceed 95% accuracy while being useless, and that the loss function must be derived from the measured positive ratio rather than chosen by habit.

**Approx. Hours:** 20

**Next Week Plan:** Add user-supplied image pairs and begin live satellite integration.

---

## Week 8 — User Uploads, Copernicus Integration and AOI Workflow
**Date Range:** 21 Sep 2026 – 27 Sep 2026

**Objectives**
1. Allow users to supply their own before/after image pair.
2. Integrate the Copernicus Data Space Ecosystem for live Sentinel-2 imagery.
3. Build the area-of-interest workflow on the map.

**Work Done**
- Added custom T1/T2 upload to the Stage 2 workbench, with validation that both images share identical dimensions since the model requires co-registered inputs.
- Fixed a Windows console encoding failure that crashed the server when non-ASCII characters were printed.
- Implemented OAuth2 client-credential authentication against the Copernicus identity endpoint, with the access token cached and refreshed before expiry.
- Implemented `/api/sentinel/search` to query the `sentinel-2-l2a` STAC catalogue filtered by bounding box, date range and maximum cloud cover, returning acquisitions in chronological order.
- Implemented `/api/sentinel/preview` to render true-colour RGB previews for a selected acquisition through the Sentinel Hub Process API using a custom evalscript.
- Built the Leaflet GIS view where the user draws a rectangle to define the AOI, converted into a `[west, south, east, north]` bounding box for the backend.
- Added the historical timeline panel so a user can select a "FROM" and a "TO" acquisition.
- Began converting project files to TypeScript for stronger typing in the frontend.

**Tools & Technologies:** Copernicus Data Space Ecosystem, Sentinel Hub Process API, STAC, OAuth2, Python `requests`, Leaflet, React.

**Outcome / Deliverable:** A working live-data path: draw an AOI → search real Sentinel-2 acquisitions → preview imagery → select a date pair.

**Challenges & Resolution**
- *Invalid bounding boxes near the antimeridian:* boxes crossing ±180° produced rejected queries. Added a longitude-wrapping helper and explicit validation before any API call.
- *Credential sprawl:* credentials were initially read in several places. Consolidated them into a single loader reading `SENTINEL_CLIENT_ID` and `SENTINEL_CLIENT_SECRET` from `.env`, confirmed `.env` stayed out of version control, and documented the keys in `.env.example`.
- *Preview latency:* large requested images were slow, so pixel dimensions were capped with sensible defaults.

**Learning Outcome:** Gained practical experience with OAuth2-protected geospatial APIs, STAC catalogue querying, and the distinction between a catalogue record and a rendered raster.

**Approx. Hours:** 20

**Next Week Plan:** Add visual super-resolution, quantify cloud contamination inside the AOI, and document the system formally.

---

## Week 9 — Super-Resolution, Cloud Analytics and Documentation
**Date Range:** 28 Sep 2026 – 04 Oct 2026

**Objectives**
1. Add 4× super-resolution for visual inspection.
2. Compute cloud statistics inside the user's AOI rather than relying on tile-wide values.
3. Produce formal technical documentation.

**Work Done**
- Integrated the SwinIR-M classical super-resolution network (`src/swinir/network_swinir.py`, 11.9M parameters) with an inference wrapper handling reflection padding so arbitrary input sizes upscale cleanly by 4×.
- Added `/api/sentinel/superresolve` and a status endpoint that reports checkpoint availability, so the interface can disable the feature gracefully rather than failing at click time.
- Kept SwinIR deliberately **isolated** from the segmentation pipeline — it upscales for display only and is never fed into either U-Net, because it synthesises detail rather than measuring it.
- Implemented `/api/sentinel/cloud-stats`, requesting the 20 m Scene Classification Layer over the AOI and classifying SCL classes 3, 7, 8, 9 and 10 as cloud or obscured, returning cloud and clear percentages plus a per-class breakdown.
- Added a cyan translucent cloud overlay on the map so an analyst can see obscured pixels before running inference.
- Set the map to open at a global view instead of a fixed regional centre.
- Wrote `PROJECT_REPORT.md` as a formal engineering report and expanded `README.md` with architecture diagrams, measured-results tables and an implemented-versus-planned matrix.

**Tools & Technologies:** SwinIR, timm, PyTorch, Sentinel-2 SCL band, Leaflet overlays, Mermaid diagrams, Markdown.

**Outcome / Deliverable:** 4× visual super-resolution and AOI-specific cloud quantification, plus complete project documentation.

**Challenges & Resolution**
- *Colour shift after upscaling:* early super-resolved outputs appeared tinted. Traced to inconsistent channel-order and value-range handling between PIL and tensor conversion, and fixed it with a dedicated diagnostic script that compared inputs and outputs side by side.
- *Conflicting cloud figures:* scene-level `eo:cloud_cover` often disagreed with local conditions. Resolved by computing statistics from the SCL band clipped to the AOI and labelling clearly in the interface which figure is being shown.

**Learning Outcome:** Learned to distinguish a visual aid from a measurement — upscaling improves inspection but adds no real sensor information — and that cloud assessment is only meaningful at the AOI scale.

**Approx. Hours:** 18

**Next Week Plan:** Consolidate branches, complete the evaluation runs, and demonstrate the full system.

---

## Week 10 — Consolidation, Evaluation and Milestone Demonstration
**Date Range:** 05 Oct 2026 – 11 Oct 2026

**Objectives**
1. Merge the diverged branches into a single consistent codebase.
2. Run and record the final benchmark evaluations for both stages.
3. Demonstrate the complete system end to end.

**Work Done**
- Reconciled several branches and pull requests covering the Stage 2 work, the GIS interface and the frontend theme, resolving conflicts in the Flask application and the React entry points.
- Re-ran the application end to end after merging to confirm no regressions.
- Ran the Stage 1 evaluation over the 15-image validation split at threshold 0.5, producing `outputs/evaluation_metrics.json` and `evaluation_report.txt`: IoU 0.8386, Dice 0.9122, precision 0.9221, recall 0.9025, pixel accuracy 0.9093.
- Ran the Stage 2 held-out test evaluation over all 33 test pairs, producing `outputs/temporal/temporal_test_results.json` and `temporal_test_report.txt`: global IoU 0.3828, Dice 0.5536, precision 0.6412, recall 0.4871, pixel accuracy 0.9632, with TP 49,330, FP 27,602, FN 51,944 and TN 2,033,812.
- Generated prediction visualisations, overlay grids and training-curve plots under `outputs/` so every reported figure traces back to an artefact.
- Fixed seeds at 42 for dataset splitting and augmentation, and standardised the evaluation threshold at 0.5 so results are comparable across runs.
- Demonstrated the full workflow: draw an AOI, select two Sentinel-2 dates, preview imagery, and run Stage 2 inference to obtain a change mask with area in hectares.
- Presented the milestone to the guide and recorded feedback on frontend polish and live-inference integration.

**Tools & Technologies:** Git branching and merging, GitHub pull requests, PyTorch, NumPy, Matplotlib, Flask, React.

**Outcome / Deliverable:** A single consistent codebase with reproducible evaluation artefacts and a demonstrated end-to-end system.

**Challenges & Resolution**
- *Conflicting model loaders:* two branches loaded the Stage 2 checkpoint differently. Unified the loader with the `cu*` → `conv_up*` remapping and made it infer the input channel count from the checkpoint, so both 6-channel and 9-channel checkpoints load correctly.
- *Reproducibility gaps:* early runs were not seeded consistently, so numbers drifted between attempts. Fixed seeds and threshold, then regenerated all reported metrics from committed scripts.
- *Scope pressure:* remaining ideas exceeded the diary period. Prioritised the live-inference integration and interface redesign for the following weeks, and documented them as planned work rather than claiming them as complete.

**Learning Outcome:** Learned how costly unmanaged branching is in a small team, and that an evaluation figure is only trustworthy when it is produced by a committed script with fixed seeds and a fixed threshold.

**Approx. Hours:** 18

---

## Appendix — Work Continued After the Diary Period (12–27 Oct 2026)

Recorded here for completeness; this work falls outside the official ten-week window.

**12–18 Oct — Error analysis and limitations**
Examined the Stage 2 confusion matrix and found roughly as many false negatives (51,944) as true positives (49,330), meaning the model detects about half of real forest loss. Documented why 96% pixel accuracy is misleading at 4.7% positive prevalence, the domain gap between the benchmark crops and live Sentinel-2 L2A imagery, the absence of cloud masking during inference, and the fact that SwinIR is a visualisation tool only.

**19–25 Oct — Build stability and cross-sensor pipeline**
Fixed a frontend build crash so `npm run build` completes cleanly. Added an "Amazon" preset button that jumps the map to a known deforestation region. Wrote `src/sen2naip_dataset.py` pairing Sentinel-2 L2A patches (10 m, 121×121) with NAIP imagery (2.5 m, 484×484), handling int32 reflectance scaling and cropping to 120/480 so the 8×8 SwinIR attention windows divide evenly. Added dataset preparation and training scripts plus a Colab notebook.

**26 Oct–01 Nov — Live inference and interface redesign**
Fixed the Sentinel-2 timeline by following STAC pagination to exhaustion and enforcing a strict chronological sort. Redesigned the interface into five views: GIS Map, Stage 1 Workbench, Stage 2 Workbench, Temporal Swipe Viewer and Live Analytics. Connected live Sentinel-2 selections directly to `/api/sentinel/analyze`, which builds `[T1, T2, |T1 − T2|]` from the two acquisitions and returns a change overlay with area in hectares, carrying an explicit domain-gap disclosure.

---

## Consolidated Results for the Report

### Stage 1 — Forest Cover Segmentation (15 validation images, threshold 0.5)

| Metric | Value |
|---|---|
| IoU (Jaccard) | 0.8386 |
| Dice (F1) | 0.9122 |
| Precision | 0.9221 |
| Recall | 0.9025 |
| Pixel accuracy | 0.9093 |
| Best epoch | 5 |

### Stage 2 — Bi-Temporal Deforestation Detection (33 held-out test pairs, threshold 0.5)

| Metric | Value |
|---|---|
| Global IoU | 0.3828 |
| Global Dice (F1) | 0.5536 |
| Global precision | 0.6412 |
| Global recall | 0.4871 |
| Pixel accuracy | 0.9632 |
| TP / FP / FN / TN | 49,330 / 27,602 / 51,944 / 2,033,812 |
| Predicted vs ground-truth change | 3.56% vs 4.68% |

### Baseline Comparison

| Approach | Test IoU |
|---|---|
| Post-classification comparison (subtract two independent masks) | 0.0026 |
| Direct 9-channel Temporal U-Net | 0.3828 |

---

## Skills Demonstrated Across the Ten Weeks

- **Deep learning:** semantic segmentation, bi-temporal change detection, U-Net architectures, class-imbalance mitigation with weighted BCE, focal and Dice losses, transfer of pretrained super-resolution models, controlled experiment design, reproducible evaluation.
- **Geospatial engineering:** OAuth2 API integration, STAC catalogue querying, Sentinel Hub Process API evalscripts, SCL-based cloud assessment, ground-sample-distance area computation, coordinate validation.
- **Web development:** Flask REST API design, React single-page application, Leaflet map interaction and overlays, Vite build tooling, multipart uploads, base64 image transport.
- **Engineering practice:** Git branching and pull-request workflow, artefact-based reporting, honest limitations analysis, scripted verification.

---

*Note: adjust the approximate hours and dates to match your institution's diary format and your actual submission schedule.*
