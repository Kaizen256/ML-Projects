# LeNet-5 (1998) – NumPy Implementation from Scratch

In this notebook I build LeNet-5, the CNN introduced by Yann LeCun for handwritten digit recognition.

## Dataset

- **Dataset**: MNIST (60k train / 10k test)  
- **Input size**: LeNet expects 32×32 greyscale images.  
  MNIST is 28×28, so we pad 2 pixels on each side in the first convolution layer.

## Architecture

| Layer      | Type                 | Parameters                             | Output Shape (input 28x28) |
| ---------- | -------------------- | -------------------------------------- | -------------------------- |
| **C1**     | Convolution          | 6 filters, 5×5 kernel, stride=1, pad=2 | (6, 28, 28)                |
|            | Activation (Sigmoid) |                                        | (6, 28, 28)                |
| **S2**     | Average Pooling      | 2×2 window, stride=2                   | (6, 14, 14)                |
| **C3**     | Convolution          | 16 filters, 5×5 kernel, stride=1       | (16, 10, 10)               |
|            | Activation (Sigmoid) |                                        | (16, 10, 10)               |
| **S4**     | Average Pooling      | 2×2 window, stride=2                   | (16, 5, 5)                 |
| **C5**     | Convolution          | 120 filters, 5×5 kernel, stride=1      | (120, 1, 1)                |
|            | Activation (Sigmoid) |                                        | (120,)                     |
| **F6**     | Fully Connected      | 120 → 84                               | (84,)                      |
|            | Activation (Sigmoid) |                                        | (84,)                      |
| **Output** | Fully Connected      | 84 → 10                                | (10,)                      |

![Architecture](figures/Architecture.png)
Image source: Zhang, Aston and Lipton, Zachary C. and Li, Mu and Smola, Alexander J. - https://github.com/d2l-ai/d2l-en

## NumPy Implementation

- Loaded and preprocessed MNIST from sklearn, normalized to [0,1], reshaped to (batch, 1, 28, 28).
- Wrote explicit functions for:
  - sigmoid, sigmoid_deriv, softmax
  - CrossEntropy loss
  - padding for convolution inputs
- Implemented:
  - Convolution layer forward & backward (manual multi-channel sum with for-loops, plus gradient calc for weights & inputs).
  - Average pooling forward & backward (distributes gradients uniformly).
- Used Xavier initialization for all kernels and zeros for biases.
- Training loop with SGD, explicit forward & backward passes, data shuffling each epoch.

Even after vectorizing, the convolution’s manual sliding windows with for loops were extremely slow. After 12 hours, not even a single epoch completed on my laptops CPU.

## PyTorch Replication

To get practical results, I replicated the model in PyTorch and trained it on Kaggle’s GPUs (T4 x2).

- Normalized MNIST using mean=0.1307, std=0.3081.
- Used nn.Conv2d, nn.AvgPool2d, nn.Sigmoid, nn.Flatten, nn.Linear.
- Applied Xavier initialization and DataParallel for multi-GPU.
- Trained with SGD and CrossEntropyLoss for 10 epochs, achieving rapid convergence.

## Conclusion

✅ Successfully implemented LeNet-5 from scratch in NumPy, including full manual forward & backward passes for convolutions, pooling, and dense layers.  
✅ Replicated the architecture in PyTorch and trained it efficiently on GPU.

❌ Couldn’t fully train the NumPy model due to impractical CPU compute time, highlighting how critical optimized libraries like PyTorch are.

This project gave me deep insights into the math, data flows, and engineering trade-offs behind CNNs. Even a relatively small CNN like LeNet-5 becomes computationally heavy to train purely from scratch on CPU. This exercise was invaluable to truly understand what frameworks like PyTorch do under the hood.

Built by Kaizen Rowe