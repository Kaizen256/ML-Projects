# Character-level RNN From Scratch (NumPy)

This project implements a simple character level recurrent neural network (RNN) entirely from scratch using NumPy.  
It trains on a small corpus of Shakespeare text and generates new text in the same style.

Unlike typical machine learning projects that rely on frameworks like TensorFlow or PyTorch, this project manually constructs every part of the model from weight matrices to forward and backward propagation providing insight into the internals of RNNs.

## Features

- Builds a recurrent neural network using only NumPy.
- Implements Backpropagation Through Time with gradient clipping.
- Generates text one character at a time, based on learned probabilities.
- Uses temperature sampling to adjust creativity vs. stability in generated output.
- Fully documented with explanations for each step.

## How it works

### 1. Preprocessing

- Reads a text file (`smallShake.txt`), lowercases all characters, and replaces non alphabetic symbols with spaces.
- Builds a character vocabulary, mapping each character to an index.
- Converts the text into a list of one-hot encoded vectors.

### 2. Preparing training sequences

- Slices the text into overlapping sequences of length 20.
- Each training example is a sequence of 20 one-hot encoded characters with targets shifted by one step ahead.

### 3. RNN architecture

- A simple vanilla RNN with:
  - Input to hidden weights (wx)
  - Hidden to hidden recurrence weights (wh)
  - Hidden bias (b)
  - Hidden to output weights (wy)
  - Output bias (by)

### 4. Training

- Uses Backpropagation Through Time to compute gradients across the entire sequence.
- Applies gradient clipping to avoid exploding gradients.
- Applies early stopping to stop training if the loss stops improving.

### 5. Text generation

- Takes a starting prompt, feeds it through the RNN to warm up the hidden state, then generates new characters one at a time.
- Uses softmax sampling with temperature, allowing control over randomness and creativity.


# Conclusion

This project demonstrates how to build a working character-level RNN entirely from scratch, understanding every detail of:
- How sequences flow through time via the hidden state.
- How backpropagation works across multiple time steps.
- How sampling with softmax temperature can control the creativity of generated text.

There were a few problems encountered along the way. At first I didn't clip the gradients so they exploded very fast. The second was that I initialized the hidden state (h) outside of the training loop. This made it so every single time a new sequence came in, h wasn't reset, so it would have a bunch of garbage values, lowering the performance of the model. This caused the model to predict random characters. When I fixed the problem and the hidden state was reset after every sequence, performance went up and generated coherent words.