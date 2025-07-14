# 🧠 Transformer-Based Neural Machine Translation 

This project implements a **Transformer model from scratch** using PyTorch to perform **Neural Machine Translation** between English and Italian. Inspired by the original ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) paper, this implementation includes custom modules for attention, positional encoding, and layer normalization — without relying on high-level libraries like Hugging Face Transformers.

---

## 🚀 Features

- Complete Transformer architecture (encoder-decoder) built from the ground up
- Custom:
  - Multi-head attention layers
  - Positional encodings
  - Residual connections and normalization
- Word-level tokenization using Hugging Face `tokenizers`
- Data loading with Hugging Face `datasets`
- Training with label smoothing and padding masking
- Evaluation metrics:
  - **BLEU Score**
  - **Word Error Rate (WER)**
  - **Character Error Rate (CER)**
- Greedy decoding logic for inference
- Attention visualization in Jupyter notebooks

---

## 🧩 Project Structure

