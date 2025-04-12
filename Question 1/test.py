import os
import pathlib
import random
import string
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import TextVectorization

# Set Keras backend
os.environ["KERAS_BACKEND"] = "tensorflow"

text_file = "English-Spanish.txt"
vocab_size = 15000
sequence_length = 20
batch_size = 64
embed_dim = 256
latent_dim = 2048
num_heads = 8
epochs = 2

# Load and preprocess data
def load_text_pairs(filepath):
    with open(filepath, encoding='utf-8') as f:
        lines = f.read().strip().split("\n")
    text_pairs = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 2:
            eng = parts[0]
            spa = f"[start] {parts[1]} [end]"
            text_pairs.append((eng, spa))
    return text_pairs

# Train/val/test split
def split_data(pairs, val_fraction=0.15):
    random.shuffle(pairs)
    num_val_samples = int(val_fraction * len(pairs))
    num_train_samples = len(pairs) - 2 * num_val_samples
    return (
        pairs[:num_train_samples],
        pairs[num_train_samples:num_train_samples + num_val_samples],
        pairs[num_train_samples + num_val_samples:]
    )

text_pairs = load_text_pairs(text_file)
train_pairs, val_pairs, test_pairs = split_data(text_pairs)


# Strip characters for custom standardization
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

def custom_standardization(input_string):
    lowercase = tf_strings.lower(input_string)
    # Keep digits and /
    return tf_strings.regex_replace(lowercase, "[^a-z0-9/]", "")

eng_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
spa_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    # Remove the standardize argument
)
train_eng_texts = [pair[0] for pair in train_pairs]
train_spa_texts = [pair[1] for pair in train_pairs]
eng_vectorization.adapt(train_eng_texts)
spa_vectorization.adapt(train_spa_texts)

def format_dataset(eng, spa):
    eng = eng_vectorization(eng)
    spa = spa_vectorization(spa)
    return (
        {
            "encoder_inputs": eng,
            "decoder_inputs": spa[:, :-1],
        },
        spa[:, 1:],
    )


def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.cache().shuffle(2048).prefetch(16)


train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

# -----------------------------
# Transformer Components
# -----------------------------

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(0, length, 1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

# ENCODER
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = tf.keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, None, :], dtype="int32")
        else:
            padding_mask = None

        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


# DECODER
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = tf.keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        # Create a look-ahead mask for self-attention (decoder)
        seq_len = tf.shape(inputs)[1]
        look_ahead_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=look_ahead_mask
        )
        out1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=None
        )
        out2 = self.layernorm_2(out1 + attention_output_2)

        proj_output = self.dense_proj(out2)
        return self.layernorm_3(out2 + proj_output)


# -----------------------------
# Build Model
# -----------------------------
# Encoder part
encoder_inputs = tf.keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
encoder = tf.keras.Model(encoder_inputs, encoder_outputs)

# Decoder part
decoder_inputs = tf.keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
decoder_outputs = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoder_outputs)

# Final output layer (softmax over vocab)
final_output = layers.Dense(vocab_size, activation="softmax")(decoder_outputs)

# Build final model
transformer = tf.keras.Model([encoder_inputs, decoder_inputs], final_output)


# -----------------------------
# Train Model
# -----------------------------

transformer.summary()
transformer.compile(
    "rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
transformer.fit(train_ds, epochs=epochs, validation_data=val_ds)

# -----------------------------
# Inference
# -----------------------------

# Define the decoding function
def decode_sequence(input_sentence):
    # Tokenize the input sentence (English)
    encoder_input = eng_vectorization([input_sentence])

    # Start with the index for “[start]”
    start_index = spa_vectorization(["[start]"]).numpy()[0][0]
    end_index = spa_vectorization(["[end]"]).numpy()[0][0]

    decoded_indices = [start_index]

    for _ in range(max_decoded_sentence_length):
        # Convert decoded indices into a tensor
        decoder_input = tf.convert_to_tensor([decoded_indices])
        
        # Get predictions
        predictions = transformer([encoder_input, decoder_input])
        
        # Predict the next token (last timestep)
        next_token_index = tf.argmax(predictions[0, -1, :]).numpy()
        decoded_indices.append(next_token_index)

        # Stop if we hit the end token
        if next_token_index == end_index:
            break

    # Convert indices back to tokens
    translated_tokens = [spa_index_lookup.get(i, "") for i in decoded_indices[1:-1]]  # exclude [start] and [end]
    return " ".join(translated_tokens)

# Define the maximum length for decoded sentences
max_decoded_sentence_length = 20

# Define spa_index_lookup
spa_vocab = spa_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))

# Define the English sentence to translate
English_sentence1 = "Deep Learning is widely used in Natural Language Processing, as Dr. Sun said in CSC 446/646."
English_sentence2 = "Natural language is how humans speak and write."
translated_sentence = decode_sequence(English_sentence1)
print("Translated Sentence:", translated_sentence)
translated_sentence = decode_sequence(English_sentence2)
print("Translated Sentence:", translated_sentence)