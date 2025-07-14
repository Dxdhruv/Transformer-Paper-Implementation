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

📦 Requirements
Python 3.8+
PyTorch
transformers, datasets, tokenizers
torchmetrics
tensorboard
tqdm

pip install -r requirements.txt

🏋️‍♂️ Training
python train.py

This will:
Train the Transformer model from scratch
Save checkpoints to the weights/ directory
Log training loss and validation metrics to TensorBoard

🔍 Validation
At the end of each epoch, the model runs greedy decoding and prints:
Source sentence
Ground truth translation
Model-predicted translation

📊 Metrics
Validation includes:
BLEU Score: Translation quality
WER: Word-level error rate
CER: Character-level error rate

🎨 Attention Visualization
Explore attention patterns using:
attention_visualization.ipynb

🤖 Inference
You can try the model manually via:
inference.ipynb
Just load the model checkpoint and provide an English sentence — it’ll generate the Italian translation using greedy decoding.

📌 TODO
 Add beam search for better decoding quality
 Export to ONNX or TorchScript for production use
 UI / API interface for translation
 CLI support for translating text files
