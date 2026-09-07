# Brain Tumor MRI Classification

Classify brain MRI scans as **glioma**, **meningioma**, **pituitary**, or **notumor**, with model explainability through Grad-CAM attention maps.

## Problem

This project investigates four-class brain tumor classification from MRI scans. The first goal is to classify each scan as glioma, meningioma, pituitary tumor, or no tumor. The second goal is to approximately localize the visual evidence used by the model with Grad-CAM. There are no true pixel-level tumor segmentation masks in this dataset, so localization is an attention-based Option A analysis rather than a segmentation result.

## Dataset

The project uses the Kaggle Brain Tumor MRI dataset with four balanced classes:

| Split    | Glioma | Meningioma | Notumor | Pituitary |
| -------- | -----: | ---------: | ------: | --------: |
| Training |  1,400 |      1,400 |   1,400 |     1,400 |
| Testing  |    400 |        400 |     400 |       400 |

The `Testing` folder was held out and left untouched until the final evaluation. The full workflow is documented in [MRI SCAN.ipynb](MRI%20SCAN.ipynb).

## Exploratory Data Analysis

- **Data integrity:** 0 corrupt files across all 7,200 images.
- **Dimension variance:** 40 unique image sizes in the inspected sample, ranging from `173 x 201` to `1335 x 1302`.
- **Grayscale confirmation:** all sampled RGB channels were identical, confirming grayscale images stored as three-channel files.
- **Background finding:** near-black pixel fraction differed by class. `notumor` had the lowest and most variable black-pixel fraction, while glioma and meningioma generally had the highest fractions. This was the hypothesis-setting observation: background and framing might become a shortcut or shift-related signal for the classifier.

## Preprocessing

Each image was processed with:

1. Gaussian blur
2. A fixed intensity threshold selected from the EDA diagnostics
3. Morphological opening and closing
4. External contour detection with a minimum-area filter
5. Cropping to the largest valid contour
6. Resizing to `224 x 224`

Two edge cases were explicitly debugged:

- **Near-full-image bounding boxes:** 516 of 7,200 images produced a valid bounding box covering nearly the full image. The per-class rates were `notumor 30.83%`, `pituitary 26.94%`, `meningioma 7.61%`, and `glioma 3.72%`. These were valid minimal-crop cases, not failed detections, and were tracked separately from true fallbacks.
- **Noisy backgrounds:** blur plus minimum contour-area filtering removed small disconnected components that previously caused unstable crops. The notebook includes before/after threshold and crop diagnostics in the preprocessing section.

![Fixed-threshold crop comparison](preprocessing_diagnostics/crop_comparison.png)

## Models and Test Results

The baseline model is a four-block grayscale CNN with batch normalization, pooling, global average pooling, and a softmax classifier. The transfer-learning model uses an ImageNet-pretrained EfficientNetB0: first the backbone is frozen, then the final 30 backbone layers are fine-tuned with a lower learning rate.

| Model                       | Test Accuracy | Test Macro F1 |
| --------------------------- | ------------: | ------------: |
| Baseline CNN                |        91.13% |        0.9089 |
| EfficientNetB0 (fine-tuned) |        91.37% |        0.9115 |

The results are near-parity: transfer learning did not clearly beat the baseline on this dataset. That is plausible because ImageNet features were learned from natural RGB images, while these inputs are grayscale medical scans with different texture and contrast statistics. The exact values are stored in [test_model_comparison.csv](test_model_comparison.csv).

## Reproducibility

Early experiments showed run-to-run variance from unseeded initialization and augmentation. This was fixed with `tf.keras.utils.set_random_seed(42)`, together with fixed dataset split and shuffle seeds.

## Error Analysis

### Meningioma Grad-CAM

For validation-time meningioma cases, the average fraction of Grad-CAM energy inside the thresholded brain interior was **79.0%** for correctly classified cases and **66.3%** for misclassified cases. The corresponding average border energy was **17.7%** versus **28.7%**. This supports a relationship between meningioma errors and attention shifted toward the image border, while remaining an attention proxy rather than anatomical localization.

The underlying values come from [meningioma_gradcam_energy_summary.csv](gradcam_figures/meningioma_gradcam_energy_summary.csv).

![Misclassified meningioma Grad-CAM example 1](./gradcam_figures/misclassified_meningioma_0069_additional_misclassified_meningioma_1.png)

![Misclassified meningioma Grad-CAM example 2](./gradcam_figures/misclassified_meningioma_0146_additional_misclassified_meningioma_2.png)

![Misclassified meningioma Grad-CAM example 3](./gradcam_figures/misclassified_meningioma_0181_additional_misclassified_meningioma_5.png)

### Glioma Test-Time Generalization Gap

The baseline history ended at **97.54% training accuracy** and **95.80% validation accuracy**, but baseline glioma recall on the untouched test set was only **75.75%**. This points to a combination of a modest train/validation overfit and a meaningful train/test generalization gap, rather than validation performance alone explaining the weakness.

The test confusion patterns were:

- Baseline CNN: glioma -> meningioma `36`, glioma -> `notumor` `49`, glioma -> pituitary `12`.
- Fine-tuned EfficientNetB0: glioma -> meningioma `66`, glioma -> `notumor` `26`, glioma -> pituitary `4`.

The training/testing visual comparison suggests differences in framing, slice orientation, contrast, and lesion presentation. This shifted the weak-class finding from a validation-time attention pattern to a test-time generalization question: some glioma lesions may be small or subtle, while some cases may also differ in presentation from the training distribution.

![Glioma training versus testing samples](./test_glioma_diagnostics/glioma_training_testing_comparison.png)

Grad-CAM panels for EfficientNet glioma cases predicted as `notumor` are saved here:

![Glioma predicted as notumor, Grad-CAM 1](./test_glioma_diagnostics/glioma_notumor_gradcam_1.png)

![Glioma predicted as notumor, Grad-CAM 2](./test_glioma_diagnostics/glioma_notumor_gradcam_2.png)

![Glioma predicted as notumor, Grad-CAM 3](./test_glioma_diagnostics/glioma_notumor_gradcam_3.png)

## Limitations

- Grad-CAM is a coarse attention proxy, not true pixel-level segmentation.
- The dataset has no true tumor masks, so localization is approximate.
- The train/test visual comparison uses a small descriptive sample and is not a formal statistical distribution-shift test.
- The dataset was assembled from multiple sources, so inconsistencies between the `Training` and `Testing` folders are possible.

## Next Steps

Extend the Grad-CAM analysis across more glioma errors, evaluate true tumor segmentation with a dataset such as BraTS, and perform seeded repeated runs to quantify remaining reproducibility variance.
