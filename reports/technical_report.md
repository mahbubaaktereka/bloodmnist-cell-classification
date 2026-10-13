# Classifying Blood Cell Images with a Small CNN: Honest Baselines and Seed Variation on BloodMNIST

Author: Mst Mahabuba Akter Eka | October 2026 | Code and figures: github.com/mahbubaaktereka/bloodmnist-cell-classification

## Abstract

I classified 28 x 28 images of single blood cells into 8 types using the BloodMNIST dataset. A small convolutional neural network (CNN, 61,032 parameters) was compared with three baselines (majority class, logistic regression and random forest on raw pixels). All choices were made on the validation set, and the test set was used once at the end. Over 5 random seeds the CNN reached a test macro-F1 of 0.901 +/- 0.008 and accuracy of 0.916 +/- 0.006, compared with 0.809 and 0.837 for the random forest. The weakest classes were basophil, monocyte and immature granulocytes, which were the classes most often confused, mainly with immature granulocytes. This is a benchmark exercise on public data, not a clinical tool.

## 1. Introduction

Counting and recognising the types of white blood cells is a routine task in a hematology laboratory. Automatic classification of cell images is a standard machine learning problem. My aim was not to reach the highest score, but to run a small, honest experiment: build simple baselines first, check how much a CNN improves on them, report variation between random seeds, and avoid using the test set for any decision.

## 2. Data

I used the BloodMNIST subset of MedMNIST v2 [1, 2]. It has 17,092 RGB images of size 28 x 28 in 8 classes (basophil, eosinophil, erythroblast, immature granulocytes, lymphocyte, monocyte, neutrophil, platelet), split into train 11,959, validation 1,712 and test 3,421 images. The license is CC BY 4.0. The original images [3] are 360 x 363 pixels, taken from normal cells of individuals without infection or hematologic disease, with one analyzer at one hospital laboratory; MedMNIST provides them downsampled to 28 x 28.

The classes are moderately imbalanced: neutrophil is the largest (19.5%), lymphocyte and basophil are the smallest (7.1% each), with the same proportions in all splits. Because of this imbalance I report macro-F1 (the average of per-class F1 scores) next to accuracy.

## 3. Methods

**Protocol.** Every choice (number of epochs, best epoch) was made on the validation set. The test set was loaded only after the final models were saved, and it was used once.

**Baselines.** Majority class; logistic regression (max_iter = 100, it did not fully converge); random forest (200 trees, random_state = 42). Images were flattened to 2,352 values and divided by 255. Each baseline was fitted once.

**CNN.** Three convolution layers (16, 32, 64 filters, 3 x 3, ReLU, max-pooling), then two fully connected layers (576 -> 64 -> 8). Adam optimizer, learning rate 1e-3, batch size 128, no data augmentation, inputs normalised with mean 0.5 and standard deviation 0.5. For each run the epoch with the best validation macro-F1 was kept.

**Seeds and epochs.** The CNN was trained with seeds 0-4. I compared 15 and 30 epochs on validation with a rule fixed beforehand: count the gain only if 30 epochs is better in most seeds and the mean gain is larger than the seed standard deviation of the 15-epoch runs (0.008). With 15 epochs the validation macro-F1 was 0.871 +/- 0.008; with 30 epochs 0.908 +/- 0.006, better in 5 of 5 seeds (mean gain 0.037), so 30 epochs was used. The 30-epoch models were then trained again with their best-epoch weights saved (validation macro-F1 0.912 +/- 0.005) and these weights were used for the test set. The small difference between the two 30-epoch runs (0.908 and 0.912) comes from GPU non-determinism.

## 4. Results

Test set (3,421 images). The CNN row is mean +/- standard deviation over 5 seeds.

| Model | Test accuracy | Test macro-F1 |
|---|---|---|
| Majority class | 0.195 | 0.041 |
| Logistic regression (pixels) | 0.778 | 0.748 |
| Random forest (pixels) | 0.837 | 0.809 |
| CNN (5 seeds) | 0.916 +/- 0.006 | 0.901 +/- 0.008 |

On validation, the baselines scored macro-F1 of 0.041, 0.782 and 0.824; their test scores are slightly lower for the two trained baselines. The CNN went from 0.912 (validation) to 0.901 (test). The weakest CNN seed on the test set (macro-F1 0.887) is still above the random forest.

Per-class test F1 of the CNN (mean +/- std over seeds): platelet 0.998 +/- 0.002, eosinophil 0.981 +/- 0.003, neutrophil 0.964 +/- 0.004, erythroblast 0.927 +/- 0.008, lymphocyte 0.887 +/- 0.008, monocyte 0.824 +/- 0.024, immature granulocytes 0.823 +/- 0.011, basophil 0.801 +/- 0.028.

The largest confusions (share of the true class, 5 seeds summed) were monocyte predicted as immature granulocytes (14.0%), basophil as immature granulocytes (12.7%), lymphocyte as immature granulocytes (6.1%), basophil as monocyte (6.0%), erythroblast as immature granulocytes (4.5%) and immature granulocytes as monocyte (4.5%). See `figures/confusion_matrix_test.png`.

## 5. Discussion

The CNN improved macro-F1 by about 0.09 over the random forest and by about 0.15 over logistic regression on the test set, and this holds for every seed. Platelet, eosinophil and neutrophil are almost solved at this resolution. Errors are concentrated in a few classes, and four of the six largest confusions end in the immature granulocytes class. I did not investigate why; the source paper describes this group as a mix of several maturation stages, but I did not test whether this explains the confusions.

Seed variation was visible (test macro-F1 from 0.887 to 0.905), so a single run would have been a misleading summary. The validation-to-test drop is small, and it is expected because the epoch was chosen on validation.

## 6. Limitations

- The images are only 28 x 28 pixels, so fine details are lost.
- One dataset from one laboratory and one analyzer, with normal cells only; performance elsewhere is unknown.
- Baselines were not tuned and were fitted once (no variation estimate); logistic regression did not converge.
- For most final seeds the best epoch was 29 or 30, so training was probably not finished. I did not train longer, to keep the rule of no tuning after seeing test results.
- Only one small architecture without augmentation was tested.
- Per-class test results were computed after the overall test result, from the same saved weights; nothing was changed afterwards.
- This is a benchmark exercise and must not be used for diagnosis.

## 7. Conclusion

A small CNN clearly beats simple pixel baselines on BloodMNIST, with a test macro-F1 of 0.901 +/- 0.008 over 5 seeds. The remaining errors are concentrated in basophil, monocyte and immature granulocytes. Future work: longer training, augmentation and other architectures, larger image sizes, multiple seeds for the baselines, inspecting the confused images, and testing on another blood cell dataset.

## References

[1] Yang J, Shi R, Wei D, Liu Z, Zhao L, Ke B, Pfister H, Ni B. MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data, 10(1):41, 2023.

[2] Yang J, Shi R, Ni B. MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis. IEEE 18th International Symposium on Biomedical Imaging (ISBI), pp. 191-195, 2021.

[3] Acevedo A, Merino A, Alferez S, Molina A, Boldu L, Rodellar J. A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. Data in Brief, 30:105474, 2020. doi:10.1016/j.dib.2020.105474.
