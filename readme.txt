# 🤖 NLP Transformer Project: Machine Translation & Joke Generation

This project showcases two fundamental applications of Transformer-based models in Natural Language Processing (NLP):

1. **Machine Translation** – A custom-built Transformer model implemented in TensorFlow to translate English to Spanish.
2. **Language Modeling** – A GPT-2 model fine-tuned using Hugging Face Transformers on a large corpus of short jokes to generate humorous text.

---

## 📌 Project Objectives

- Understand and implement Transformer architectures for sequence modeling tasks.
- Build an encoder-decoder translation model from scratch in TensorFlow.
- Fine-tune a pre-trained GPT-2 model for creative language generation.
- Evaluate model performance using training metrics and output quality.

---

## 🧠 Part 1: Machine Translation

- Implemented a Transformer encoder-decoder from scratch.
- Used a dataset of 130k+ English-Spanish sentence pairs.
- Trained over 10 epochs using teacher forcing and greedy decoding.
- Evaluated based on syntactic fluency and grammar in translated output.

---

## 😂 Part 2: Joke Generation with GPT-2

- Fine-tuned `GPT2LMHeadModel` using Hugging Face Transformers.
- Trained on 200k+ short jokes.
- Generated text using custom prompts like “Why did the chicken…” and “My dog…”.
- Model output evaluated qualitatively based on humor structure and coherence.

---

## 📁 Files & Structure

