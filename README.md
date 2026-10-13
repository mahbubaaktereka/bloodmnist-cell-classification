# BloodMNIST Cell Classification

Classifying blood cell images into 8 types with a small CNN, compared against simple baselines on raw pixels.

## Problem

Given a 28x28 RGB image of a single blood cell, predict its type (8 classes: basophil, eosinophil, erythroblast, immature granulocytes, lymphocyte, monocyte, neutrophil, platelet). This is a standard benchmark task, not a clinical tool.

## Objective

1. Build honest baselines (majority class, logistic regression and random forest on raw pixels).
2. Train a small convolutional neural network (CNN) and check how much it improves over the baselines.
3. Choose every setting on the validation set only, and use the test set once at the end.

## Dataset

- MedMNIST v2, BloodMNIST subset (https://medmnist.com), license CC BY 4.0.
- 17,092 images of size 28x28x3, 8 classes.
- Split: train 11,959 / validation 1,712 / test 3,421.
- Class balance: neutrophil is the largest class (19.5%); lymphocyte and basophil are the smallest (7.1% each). Proportions are the same in all three splits.
- Loaded with the `medmnist` Python package (version 3.0.2).

## Tools

Python, PyTorch, scikit-learn, NumPy, pandas, matplotlib, the `medmnist` package. Run on Kaggle notebooks with a GPU.
