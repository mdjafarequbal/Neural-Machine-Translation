# Neural Machine Translation (NMT) - English to French 🇬🇧➡️🇫🇷

This project implements a basic **Neural Machine Translation (NMT)** model using TensorFlow and NumPy. It demonstrates how to train an encoder-decoder model that translates simple English phrases into French.

---

## 📖 Overview

This notebook demonstrates:
- Tokenizing and preparing sequence data
- Building an encoder-decoder architecture with LSTM layers
- Training the model on paired English-French sentences
- Using the trained model for inference to translate new inputs

---

## 🧠 Model Architecture

The model follows the standard sequence-to-sequence structure:

- **Encoder**: Embeds the English sentence into a context vector using LSTM.
- **Decoder**: Takes this context and generates the French sentence.

```python
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
