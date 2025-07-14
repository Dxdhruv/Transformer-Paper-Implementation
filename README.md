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
├── config.py # Project configuration
├── dataset.py # Custom Dataset class and masking logic
├── model.py # Transformer model and components
├── train.py # Training pipeline and validation logic
├── attention_visualization.ipynb # Attention heatmap visualizer
├── inference.ipynb # Inference example on test inputs
└── tokenizer_{en,it}.json # Tokenizers (generated after training starts)


---

## ⚙️ Configuration

Configuration is handled in `config.py`. Key parameters include:

```python
{
  "batch_size": 8,
  "num_epochs": 20,
  "lr": 1e-4,
  "seq_len": 350,
  "d_model": 512,
  "lang_src": "en",
  "lang_tgt": "it",
  ...
}

📦 Requirements
Python 3.8+
PyTorch
transformers, datasets, tokenizers
torchmetrics
tensorboard
tqdm

