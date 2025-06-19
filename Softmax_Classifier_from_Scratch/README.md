# Softmax Classifier on Iris Dataset (NumPy Implementation)

This project implements a minimal softmax classifier from scratch using only NumPy, trained on the classic Iris flower dataset. The model uses stochastic gradient descent (SGD) to update weights and predicts the species of an iris flower based on its petal and sepal measurements.

---

## 🧠 Model Overview

Implements a multi-class **logistic regression** model trained with:

- **Stochastic Gradient Descent**
- **Softmax activation** (to convert logits to probabilities)
- **Cross-Entropy loss** (to measure prediction error)

---

## 📊 Dataset: Iris

- **150 samples**
- **3 classes**: Setosa, Versicolor, Virginica
- **4 features**: sepal length, sepal width, petal length, petal width

Labels are one-hot encoded to allow for multiclass classification.

---

## 🚀 Results

| Metric         | Value     |
|----------------|-----------|
| Training Method| SGD       |
| Epochs         | 100       |
| Test Accuracy  | **97.37%**|

Achieved **97.37% accuracy** on the test set using NumPy-only code.

---

Built by Kaizen Rowe

