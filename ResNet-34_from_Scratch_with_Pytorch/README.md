# Tiny ImageNet Classification with Modified ResNet-34 from Scratch

This project implements a modified ResNet-34 architecture trained on the Tiny ImageNet dataset (200 classes, 64×64 images).  
It demonstrates how to adapt classical deep learning models to small-resolution image datasets while achieving competitive performance.

---

## 🚀 Project Overview

- **Dataset:** Tiny ImageNet (200 classes, 64×64 images, ~100k samples)
- **Architecture:** Modified ResNet-34
  - Changed the original 7×7 convolution to a 3×3 to better handle small images.
  - Removed the early max-pooling layer to preserve spatial detail.
- **Data augmentation:** Random cropping with padding and horizontal flipping.
- **Optimization:** SGD with momentum, weight decay, and a cosine annealing scheduler.
- **Performance:** 
  - Achieved ~61% Top-1 accuracy on the validation set after 100 epochs.

---

## 🏗️ Model Architecture

| Block    | Kernel | Stride | Padding | Output Channels | Repeats | Downsample |
| -------- | ------ | ------ | ------- | --------------- | ------- | ---------- |
| conv2_x  | 3      | 1      | 1       | 64              | 3       | No         |
| conv3_x  | 3      | 2      | 1       | 128             | 4       | Yes        |
| conv4_x  | 3      | 2      | 1       | 256             | 6       | Yes        |
| conv5_x  | 3      | 2      | 1       | 512             | 3       | Yes        |

- Starts with a 3x3 Convolutional layer with padding to maintain dimensions.
- Ends with a global average pooling, a dropout layer, and a fc layer to 200 classes.

---

## ⚙️ Training Details

| Parameter    | Value                                  |
| ------------ | -------------------------------------- |
| Optimizer    | SGD + momentum=0.9                     |
| Learning Rate| 0.01 (decayed with cosine annealing)   |
| Weight Decay | 0.0004                                 |
| Batch Size   | 64                                     |
| Epochs       | 100                                    |
| Augmentation | RandomCrop(64,4), RandomHorizontalFlip |

- Trained using CUDA on Kaggle’s P100 GPU.
- Took ~7 hours to train.
---

## 📈 Results

- Validation Top-1 Accuracy: ~61%
- Best model weights saved to `ResNet34.pth`.
- Training metrics (loss, accuracy, LR) saved in `training_history.json`.

---

## 🗂️ Files

| File                     | Description                             |
|--------------------------|-----------------------------------------|
| `ResNet-34_Scratch.ipynb`| ResNet-34 model and narrative           |
| `ResNet34_best.pth`      | Best model based on validation accuracy |
| `training_history.json`  | JSON log of train/val loss and accuracy |
| 'mean_std.py'            | Script to find mean and std of dataset  |

---

## How to Use

If you want to use the model, copy and paste the architecture of the model into your own file, then load the state dict of the trained model `ResNet34_best.pth`


model = ResNet34().to(device)
model.load_state_dict(torch.load("ResNet34_best.pth"))
model.eval()