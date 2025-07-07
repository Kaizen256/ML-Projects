# Multi-Layer Perceptron on MNIST (Numpy Implementation)

This project implements a fully connected neural network (MLP) from scratch using only Numpy, trained on the classic MNIST handwritten digits dataset. The model learns to classify digits from 0 to 9 using only fundamental linear algebra operations — no ML frameworks or libraries.

---

## 🧠 Model Overview

Implements a multi-layer neural network trained using:

- **Mini-Batch Gradient Descent** (batch size = 32)
- **ReLU activations** in the hidden layers
- **Softmax output** for probability prediction
- **Cross-Entropy loss** for classification error
- **He Initialization** for stable learning
- **Manual Backpropagation** using NumPy

---

## 📊 Dataset: MNIST

- **70,000 grayscale images** (28×28 pixels)
- **10 classes**: digits 0 through 9
- **60,000 training** and **10,000 test** images
- Input images are flattened from `28×28 → 784` features
- Labels are one-hot encoded for use with softmax

---

## ⚙️ Network Architecture
Input (784)
↓
Dense Layer (128 units) + ReLU
↓
Dense Layer (64 units) + ReLU
↓
Dense Layer (10 units) + Softmax

---

## 🚀 Results

| Metric              | Value         |
|---------------------|---------------|
| Training Method     | Mini-Batch GD |
| Epochs              | 50            |
| Final Test Accuracy | 98.31%        |

Achieved over **98% accuracy** on the MNIST test set

---

Built by Kaizen Rowe