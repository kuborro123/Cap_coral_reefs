
# 📁 Training and Evaluation Splits (YOLOv8)

This document explains the purpose and composition of each training, validation, and test split used in the coral reef segmentation experiments.

---

## 🏋️ Training Sets

### `train_balanced_1to1.txt`
- ✅ **Contains:** 200 images with only **hard coral** and 200 with **soft coral** or **mixed**.
- ⚖️ Perfectly balanced classes (1:1).
- 🔬 Good for analyzing model fairness and bias.

### `train_extreme_imbalance.txt`
- ✅ **Contains:** 1908 hard-only images and 50 soft-containing images.
- 🧪 Simulates a worst-case imbalance.
- 🎯 Purpose: test model performance under data skew.

### `train_realistic.txt`
- ✅ **Contains:** 80% of the full dataset, keeping the **natural distribution** between hard and soft coral.
- 🌍 Most representative of the real-world setting.
- ⭐ Best performance in results.

---

## 🧪 Validation Set

### `val.txt`
- ✅ **Contains:** 800 images
  - 300 soft-containing (soft-only or mixed)
  - 500 hard-only
- 🔄 Same for all training runs.
- 🎯 Used to compare model generalization consistently.

---

## 🧪 Test Set

### `test.txt`
- ✅ **Contains:** Stratified 20% of the dataset (never used in training).
- 📏 Final unbiased evaluation of model performance.
- 🚫 No data leakage.

---

## 🧠 Notes

- All splits are generated using full paths from the original `output.csv`.
- Validation and test sets remain fixed for **fair comparisons**.
