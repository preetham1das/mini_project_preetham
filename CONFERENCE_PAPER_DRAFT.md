# Conference Paper Working Draft

## A Baseline Study of U-Net-Based Forest Segmentation and Bi-Temporal Deforestation Mapping

**Authors:** [Add team member names and affiliations]  
**Target venue:** [Add conference and formatting requirements]  
**Status:** Internal working draft. Experimental claims below are limited to repository artifacts; resolve the verification items at the end before submission.

> **Scope note for the team:** This draft centers on the two offline segmentation experiments. The repository also contains a live Sentinel-2/Web GIS application, but its current live change-analysis route uses independent single-date forest predictions followed by a post-classification rule. The offline temporal benchmark uses a separately trained Temporal U-Net. Do not describe the live route as deploying the benchmark Temporal U-Net, or treat live visualizations as quantitative validation.

## Abstract

This study documents a two-task deep-learning workflow for forest mapping from satellite imagery. First, a convolutional U-Net was trained to segment forest cover from RGB images in the repository’s Amazon Forest Dataset. On its 15-image labeled validation split, the saved evaluation reports an aggregate intersection over union (IoU) of 0.8386 and Dice score of 0.9122 at a probability threshold of 0.5. Second, a U-Net-based bi-temporal model was trained on the Forest-Change dataset to predict change directly from paired acquisitions. Its nine-channel input concatenates the pre-change RGB image, post-change RGB image, and their absolute difference. On the repository’s 33-pair test split, pooled pixel counts yield an IoU of 0.3828, Dice score of 0.5536, precision of 0.6412, and recall of 0.4871. The test set contains 4.68% positive pixels, so the 0.9632 pixel accuracy should not be interpreted alone. These results document a project baseline, not evidence of geographic generalization or superiority over alternative change-detection methods. A companion Web GIS connects to Copernicus Sentinel-2 for visualization and analysis, but its live post-classification predictions have not been evaluated against ground truth. The results motivate stronger spatially controlled evaluation, per-pair performance analysis, and direct validation on Sentinel-2 data.

**Keywords:** forest monitoring; deforestation; satellite imagery; semantic segmentation; U-Net; bi-temporal change detection; class imbalance

## 1. Introduction

Satellite imagery offers a means to monitor forest cover over areas that are difficult to survey continuously on the ground. Automated mapping remains challenging: forest loss may occupy a small fraction of an image, illumination and seasonal vegetation changes can resemble land-cover change, and predictions made at two dates can accumulate classification errors.

This project explores two related but distinct tasks. The first is single-date forest-cover segmentation. The second is supervised bi-temporal change segmentation, where a model receives an image pair and predicts a binary change mask. The implementation also includes a Web GIS workflow for inspecting Sentinel-2 acquisitions, but the web application and the offline benchmark are separate inference paths and are evaluated differently.

The paper reports the saved, reproducible measurements currently present in the repository and makes their scope explicit. It does not claim that a particular input formulation is superior without a controlled ablation, that benchmark results transfer to operational Sentinel-2 scenes, or that a high pixel accuracy alone indicates reliable deforestation detection.

### Contributions

1. We document an RGB U-Net forest-cover segmentation baseline and its measured validation performance.
2. We document a supervised bi-temporal U-Net benchmark using concatenated pre-change, post-change, and absolute-difference channels, evaluated on a held-out repository test split.
3. We report class prevalence and pooled confusion counts alongside overlap and precision/recall metrics, and distinguish offline benchmark evidence from the live GIS workflow.

These are implementation and evaluation contributions; the present evidence does not establish architectural novelty.

## 2. Materials and Methods

### 2.1 Datasets and splits

Two local datasets are used by the offline experiments.

| Dataset / task | Training | Validation | Test | Images and labels |
|---|---:|---:|---:|---|
| Amazon Forest Dataset / single-date forest segmentation | 30 | 15 | 15 | Training and validation have paired TIFF images and PNG masks; the test directory contains images without labels. |
| Forest-Change / bi-temporal change segmentation | 270 pairs | 31 pairs | 33 pairs | Each pair has pre-change (`A`), post-change (`B`), and a change-label image. |

The counts above match the files currently present in the repository. The Forest-Change report describes the source tiles as 480 × 480 RGB PNGs. The data loader resizes imagery and labels to 256 × 256 before training or evaluation. Exact sensor, acquisition dates, geographic coverage, dataset citation/license, and geographic independence between the supplied splits must be documented from the authoritative dataset source before submission.

For the Amazon masks, the loader maps raw value 2 to forest and raw value 1 to non-forest. For Forest-Change, it maps every nonzero grayscale label value to positive change. This label conversion should be checked against the dataset’s official label semantics.

The Forest-Change pixel audit in the project report gives the following class balance:

| Split | Pairs | Positive change pixels | Negative pixels | Positive proportion |
|---|---:|---:|---:|---:|
| Train | 270 | 844,851 | 16,849,869 | 4.77% |
| Validation | 31 | 106,999 | 1,924,617 | 5.27% |
| Test | 33 | 101,274 | 2,061,414 | 4.68% |

The audit implies a roughly 20:1 negative-to-positive ratio. The test split contains 2,162,688 pixels in total.

### 2.2 Preprocessing and augmentation

The loaders convert images to RGB, resize them to 256 × 256 using bilinear interpolation, scale pixel values from [0, 255] to [0, 1], and convert them to channel-first tensors. Masks are resized with nearest-neighbor interpolation to preserve class labels. Training augmentation consists of random horizontal and vertical flips and rotations by multiples of 90 degrees; paired dates and their mask receive the same spatial transform. Validation and test data are not augmented.

### 2.3 Single-date forest segmentation

The first task predicts a binary forest mask from an RGB image. The model is a standard encoder-decoder U-Net with four downsampling and four upsampling stages, skip connections, and 32 base feature channels. It outputs one logit per pixel. The saved project report gives 7,763,041 trainable parameters for this model configuration. Training code uses a combined binary-cross-entropy and soft-Dice objective, Adam optimization, and a 0.5 probability threshold for reported evaluation.

The evaluation script accumulates confusion counts over the validation set and calculates metrics from those pooled counts. Thus, although the saved report uses the word “mean,” these values are pooled-pixel metrics rather than an arithmetic mean of per-image metrics. The 15 images in the Amazon test folder are unlabeled and do not support a test-set performance claim.

### 2.4 Bi-temporal change segmentation

For each paired RGB image \(I_{T_1}, I_{T_2} \in [0,1]^{3 \times H \times W}\), the nine-channel model input is:

\[
X = [I_{T_1}, I_{T_2}, |I_{T_1} - I_{T_2}|] \in [0,1]^{9 \times H \times W}.
\]

The network is a U-Net with four encoder downsamplings, a bottleneck, skip connections, and a one-channel change-logit output. The code also supports six-channel input without the explicit difference tensor, but the saved report identifies the evaluated checkpoint as the nine-channel configuration.

The training script defines experiments that vary input channels and loss, including a nine-channel focal-plus-Dice setup. Its defaults specify batch size 4, Adam with initial learning rate \(10^{-3}\), up to 30 epochs, early-stopping patience 8, and a reduce-on-plateau scheduler. The reported focal-plus-Dice loss weights both terms equally and uses focal-loss parameters \(\gamma=2\) and \(\alpha=0.75\). The reported decision threshold is 0.5.

**Checkpoint provenance requires reconciliation:** the project report attributes the saved temporal checkpoint to epoch 21 and validation IoU 0.5043. The tracked `outputs/temporal_training_history.csv` contains only ten epochs and a maximum validation IoU of 0.3975, and no corresponding `outputs/experiments` directory was found during this analysis. The test metrics below are present in the saved test report, but the training epoch, selected experiment, and validation score must be verified from the checkpoint or a matching run log before publication.

### 2.5 Evaluation metrics

For positive-class true positives (TP), false positives (FP), false negatives (FN), and true negatives (TN), the reported metrics are:

\[
\mathrm{IoU} = \frac{TP}{TP+FP+FN}, \quad
\mathrm{Dice} = \frac{2TP}{2TP+FP+FN},
\]
\[
\mathrm{Precision} = \frac{TP}{TP+FP}, \quad
\mathrm{Recall} = \frac{TP}{TP+FN}, \quad
\mathrm{Accuracy} = \frac{TP+TN}{TP+FP+FN+TN}.
\]

The temporal test metrics are calculated from confusion counts accumulated over all 33 pairs, rather than averaging pair-level scores. This should be stated whenever the numbers are quoted.

## 3. Results

### 3.1 Amazon forest-cover validation

The saved evaluation for the best U-Net checkpoint reports results on 15 labeled validation images at threshold 0.5:

| Metric | Pooled validation result |
|---|---:|
| IoU | 0.8386 |
| Dice | 0.9122 |
| Precision | 0.9221 |
| Recall | 0.9025 |
| Pixel accuracy | 0.9093 |

These measurements describe the supplied validation split only. They are not results on the 15 unlabeled images in the Amazon test folder.

### 3.2 Forest-Change temporal test

The saved test report covers 33 pairs and 2,162,688 pixels at threshold 0.5:

| Metric | Pooled test result |
|---|---:|
| IoU | 0.3828 |
| Dice | 0.5536 |
| Precision | 0.6412 |
| Recall | 0.4871 |
| Pixel accuracy | 0.9632 |

| Confusion count | Pixels |
|---|---:|
| TP | 49,330 |
| FP | 27,602 |
| FN | 51,944 |
| TN | 2,033,812 |

The ground-truth change prevalence is 4.68%; the model predicts 76,932 positive pixels compared with 101,274 ground-truth positives. The high pixel accuracy coexists with recall below 0.5 and should not be presented as evidence of high change-detection sensitivity. The confusion counts show substantial missed change as well as false alarms.

### 3.3 Post-classification comparison and operational interface

A separate saved report evaluates a post-classification comparison (PCC) result on one pair, `test_000001`, with IoU 0.0026 and recall 0.0035. This is a single-pair observation, not a pooled PCC benchmark. It is not a controlled comparison with the Temporal U-Net across the same test split, and should not be used to claim that direct temporal learning outperforms PCC.

The current Flask Sentinel-2 analysis route uses the single-date forest model independently on the two acquisitions and marks pixels predicted as forest at \(T_1\) and non-forest at \(T_2\). The repository also contains a Copernicus data workflow and SCL-based cloud statistics. No ground-truth evaluation of live Sentinel-2 predictions is reported here; benchmark IoU and Dice must not be attributed to live AOIs. The separate Kaggle Sentinel-1/Sentinel-2 late-fusion experiment report is explicitly marked “NOT TRAINED” and is excluded from the results.

## 4. Discussion

The single-date validation result indicates that the implemented U-Net can reproduce forest labels on the small labeled Amazon validation split. Its scope is limited by the split size and absence of labels for the Amazon test images. Additional geographically separated test data are needed to estimate performance beyond these samples.

For the Forest-Change task, the temporal model obtains a pooled test IoU of 0.3828 and Dice of 0.5536. Precision exceeds recall, and the model predicts fewer change pixels than appear in the ground truth. This pattern is consistent with missed change being an important error mode. The 96.32% pixel accuracy is less informative because stable pixels dominate the test set.

The explicit difference channels are a design choice, not a demonstrated improvement in this report. Establishing their value requires a controlled ablation in which six-channel and nine-channel models use identical splits, training procedures, checkpoint selection, and evaluation thresholds. The same applies to the focal-plus-Dice objective: comparison with alternative losses must be supported by results from reproducible runs.

The implementation combines offline segmentation experiments with a practical visualization interface, but the benchmark does not establish deployment readiness. Changes in sensor characteristics, atmospheric correction, resolution, season, georegistration, and clouds can all affect live predictions. The reported PCC single-pair result further illustrates why live results require their own carefully designed validation rather than extrapolation from benchmark metrics.

## 5. Limitations

1. **Small evaluation sets:** Stage 1 has 15 labeled validation images; Stage 2 has 33 test pairs. No confidence intervals, repeated seeds, or per-pair score distribution are provided.
2. **Potential split dependence:** The repository’s split names are known, but geographic separation and near-duplicate leakage have not been independently verified.
3. **Checkpoint and history mismatch:** The stated temporal checkpoint epoch and validation IoU are not corroborated by the tracked temporal training history.
4. **No controlled model ablation:** The saved evidence does not establish the benefit of the nine-channel difference input or the selected loss versus alternatives.
5. **Threshold reporting:** Results use a threshold of 0.5; sensitivity to threshold and calibration should be reported, with threshold selection confined to validation data.
6. **Label assumptions:** The temporal loader maps every nonzero label intensity to change. The official meaning of all nonzero label values must be confirmed.
7. **Domain gap:** Forest-Change benchmark results do not establish performance on live Sentinel-2 L2A imagery or on other regions, seasons, and sensors.
8. **Operational validation:** Live analysis has no reference labels, and cloud statistics do not constitute cloud masking or model validation.
9. **Dataset provenance:** Authoritative citations, license/permissions, sensor metadata, acquisition dates, and geographic coverage need to be added.

## 6. Conclusion

This project provides two measured baselines: pooled-pixel forest segmentation performance on a small Amazon validation set, and pooled-pixel bi-temporal change performance on a 33-pair Forest-Change test split. The temporal test result shows partial overlap with the ground truth but also substantial false negatives; pixel accuracy alone overstates the practical picture under strong class imbalance. The accompanying GIS application demonstrates a route to satellite-data inspection, but its current post-classification predictions are separate from the offline temporal benchmark and have not been validated against ground truth. The next step is not to broaden the claims, but to reconcile checkpoint provenance, confirm dataset and label semantics, and conduct controlled, geographically aware evaluation.

## 7. Recommended Work Before Submission

- **Reproducibility owner:** Match `best_temporal_model.pth` to a training run, experiment configuration, seed, epoch, validation score, and the test evaluation artifact. Preserve the matching history and logs.
- **Dataset owner:** Add authoritative dataset citations, licenses, sensor/date/geography details, label mapping, and evidence that train/validation/test do not share geographic blocks or near-duplicate scenes.
- **Evaluation owner:** Export per-pair IoU/Dice/precision/recall, confidence intervals (preferably resampling at the pair or geographic-block level), and threshold-sensitivity results. Keep test data out of model/threshold selection.
- **Ablation owner:** Compare six-channel versus nine-channel input, and the defined loss variants, under one fixed protocol. Include a same-test-set PCC baseline if comparative claims are intended.
- **Operational-validation owner:** If the paper discusses live Sentinel-2 analysis quantitatively, assemble georeferenced, cloud-screened, labeled Sentinel-2 pairs and evaluate the actual live inference route. Otherwise keep live GIS functionality as system context only.
- **Writing owner:** Replace bracketed metadata, verify references against the required conference style, and remove all unverified claims before submission.

## References (starter list; verify and complete)

1. Ronneberger, O., Fischer, P., and Brox, T. “U-Net: Convolutional Networks for Biomedical Image Segmentation.” *MICCAI*, 2015, pp. 234–241.
2. Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollár, P. “Focal Loss for Dense Object Detection.” *ICCV*, 2017, pp. 2980–2988.
3. Add the authoritative citation and license for the Amazon Forest Dataset used in this repository.
4. Add the authoritative citation and license for the Forest-Change dataset used in this repository.
5. Add references for remote-sensing bi-temporal change detection and spatially independent evaluation that are directly relevant to the final experimental protocol.
