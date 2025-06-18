# 🚗 Hybrid CNN + Transformer Model for Self-Driving Car Steering Angle Prediction

This project implements a hybrid deep learning architecture combining a **Convolutional Neural Network (CNN)** and a **Transformer block** to predict steering angles from road images. The model architecture is inspired by the NVIDIA self-driving car model and enhanced with attention mechanisms for better contextual understanding.

---

## 🧠 Model Architecture

- **Input Shape:** `(66, 200, 3)` RGB road images.
- **CNN Layers:** Extract spatial features (based on NVIDIA’s architecture).
- **Transformer Block:**
  - Multi-Head Self-Attention
  - Layer Normalization
  - Feed Forward Network (FFN)
- **Dense Layers:** Refine the representation.
- **Output:** Single regression output (steering angle).

---

## 🏗️ Architecture Summary

```plaintext
Input (66x200x3)
   ↓
Conv2D Layers (NVIDIA style)
   ↓
Flatten → Dense → Reshape to (1, transformer_dim)
   ↓
Transformer Block (MultiHeadAttention + FFN + Residual Connections)
   ↓
GlobalAveragePooling1D
   ↓
Dense(100) → Dropout
   ↓
Dense(50) → Dropout
   ↓
Dense(10)
   ↓
Output: Dense(1) [Steering Angle]
