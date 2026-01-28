# Vision Transformer (ViT)

## Overview

This project implements a Vision Transformer (ViT) model for image classification using TensorFlow/Keras.  
The model replaces convolutional feature extraction with transformer-based self-attention by treating an image as a sequence of fixed-size patches.

The implementation focuses on understanding the core ViT architecture rather than production optimization.

---

## How to Use

1. Install necessary dependencies using requirements txt file ( pip install -r requirements.txt )
2. Run the Streamlit UI file ( streamlit run ui.py )

---

## Vision Transformer Concept

Instead of using convolutional filters, the image is:
1. Split into fixed-size patches
2. Linearly embedded into vectors
3. Processed using transformer encoder blocks with self-attention

This allows the model to capture global relationships across the entire image from the first layer.

---

## Architecture

- Input image is divided into non-overlapping patches
- Each patch is flattened and projected into an embedding space
- Learnable positional embeddings are added
- A stack of transformer encoder blocks is applied
- A classification token is used for final prediction

---

## Transformer Encoder Block

Each encoder block contains:
- Multi-head self-attention
- Layer normalization
- Feed-forward neural network (MLP)
- Residual connections

---

## Dataset

- CIFAR-10
- 10 classes
- 32×32 RGB images

Basic preprocessing includes normalization. No external pretraining is used.

---

## Training Setup

- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Batch size: 64
- Epochs: 20

Hyperparameters such as patch size, embedding dimension, number of heads, and number of transformer layers are configurable in the code.

---

## Results

1. The model achieves reasonable performance on CIFAR-10 but does not outperform well-tuned CNNs when trained from scratch.
2. This is expected behavior, as Vision Transformers require large datasets or pretrained weights to fully realize their potential.

<img width="2560" height="1440" alt="Screenshot (6)" src="https://github.com/user-attachments/assets/d97a9c59-9af4-4568-b0a7-96a1d937ce1b" />

---

## Key Takeaways

- Vision Transformers model global context directly
- Positional embeddings are essential
- ViTs are data-hungry compared to CNNs
- Training from scratch on small datasets is suboptimal

---

## Summary

This project demonstrates the core mechanics of Vision Transformers and highlights the architectural tradeoffs involved when replacing convolutional networks with transformer-based models for vision tasks.
