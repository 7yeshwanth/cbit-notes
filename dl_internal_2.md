# ***Deep Learning LAB-INTERNAL - 2***

## **De-noising Auto-Encoder**

```py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, datasets
import matplotlib.pyplot as plt

# 1. Load and prepare data
(x_train, _), (x_test, _) = datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.
x_test = x_test.reshape(-1, 784).astype('float32') / 255.

# 2. Add noise to test images
noisy_test = x_test + 0.5 * np.random.normal(size=x_test.shape)
noisy_test = np.clip(noisy_test, 0, 1)

# 3. Simple autoencoder model
model = tf.keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(784, activation='sigmoid')
])
model.compile(optimizer='adam', loss='mse')

# 4. Train (using noisy training data)
noisy_train = x_train + 0.5 * np.random.normal(size=x_train.shape)
noisy_train = np.clip(noisy_train, 0, 1)
model.fit(noisy_train, x_train, epochs=5, batch_size=256)

# 5. Get denoised images
denoised = model.predict(noisy_test)

# 6. Show samples
plt.figure(figsize=(10, 4))
for i in range(5):  # show first 5 samples
    # Original
    plt.subplot(3, 5, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Noisy
    plt.subplot(3, 5, i+6)
    plt.imshow(noisy_test[i].reshape(28, 28), cmap='gray')
    plt.title("Noisy")
    plt.axis('off')

    # Denoised
    plt.subplot(3, 5, i+11)
    plt.imshow(denoised[i].reshape(28, 28), cmap='gray')
    plt.title("Denoised")
    plt.axis('off')

plt.tight_layout()
plt.show()
```

## **RNN**

### **RNN - Embedding (Sentiment Analysis)**
```py
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Load and preprocess data
vocab_size, max_length = 10000, 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
x_train = pad_sequences(x_train, maxlen=max_length, padding='post')
x_test = pad_sequences(x_test, maxlen=max_length, padding='post')

# Build and train model
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_length),
    SimpleRNN(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=tf.keras.optimizers.Adam(0.0005),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Predict
sample = ["this movie was fantastic"]
sample_seq = tf.keras.preprocessing.text.text_to_word_sequence(sample[0])
sample_idx = [[imdb.get_word_index().get(word, 0) for word in sample_seq if imdb.get_word_index().get(word, 0) < vocab_size]]
sample_pad = pad_sequences(sample_idx, maxlen=max_length)
print("Sentiment:", "Positive" if model.predict(sample_pad) > 0.5 else "Negative")
```

### **RNN for Sequence Labelling Problem( parts of Speech tagging)**
```py
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Sample dataset (sentences and POS tags)
texts = ["I love coding", "She is running", "Dogs are cute"]
pos_tags = [
    ["PRON", "VERB", "NOUN"],
    ["PRON", "AUX", "VERB"],
    ["NOUN", "AUX", "ADJ"]
]

# Tokenizing words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
print("Word Index:", word_index)

# Convert words to sequences
sequences = tokenizer.texts_to_sequences(texts)
print("Sequences:", sequences)

# Padding sequences to a fixed length
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', value=0)
print("Padded Sequences:\n", padded_sequences)

# POS Tag Mapping
tag_vocab = {tag: i for i, tag in enumerate(set(tag for tags in pos_tags for tag in tags))}
y = [[tag_vocab[tag] for tag in tags] for tags in pos_tags]

# Padding POS sequences
y_padded = pad_sequences(y, maxlen=max_length, padding='post')

# Convert labels to categorical format
y_categorical = np.array([to_categorical(seq, num_classes=len(tag_vocab)) for seq in y_padded])

# Model definition
model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=8, input_length=max_length),
    SimpleRNN(32, return_sequences=True),
    Dense(len(tag_vocab), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, y_categorical, epochs=100, verbose=1)

# Predict on a new sentence
def predict_pos(sentence):
    encoded_sentence = tokenizer.texts_to_sequences([" ".join(sentence)])[0]
    padded_sentence = pad_sequences([encoded_sentence], maxlen=max_length, padding='post')
    predictions = model.predict(padded_sentence)
    predicted_tags = [list(tag_vocab.keys())[np.argmax(tag)] for tag in predictions[0]]
    return list(zip(sentence, predicted_tags))

# Test prediction
print(predict_pos(["I", "am", "happy"]))

```

## **Machine Translation**
### **Machine Translation with LSTM based Encoder - Decoder (without attention machanism)**
```py
import numpy as np
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tiny dataset
eng = ["hello", "goodbye"]
hin = ["<start> नमस्ते <end>", "<start> अलविदा <end>"]

# Tokenize
eng_tok = Tokenizer()
eng_tok.fit_on_texts(eng)

hin_tok = Tokenizer(filters='')
hin_tok.fit_on_texts(hin)

# Prepare data
X = pad_sequences(eng_tok.texts_to_sequences(eng))
y = pad_sequences(hin_tok.texts_to_sequences(hin))

# Model
enc_in = Input(shape=(X.shape[1],))
enc_emb = Embedding(len(eng_tok.word_index)+1, 64)(enc_in)
enc_out, h, c = LSTM(64, return_state=True)(enc_emb)

dec_in = Input(shape=(y.shape[1]-1,))
dec_emb = Embedding(len(hin_tok.word_index)+1, 64)(dec_in)
dec_out, _, _ = LSTM(64, return_sequences=True, return_state=True)(dec_emb, initial_state=[h, c])
out = Dense(len(hin_tok.word_index)+1, activation='softmax')(dec_out)

model = Model([enc_in, dec_in], out)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit([X, y[:,:-1]], y[:,1:], epochs=300)

# Build encoder model separately
encoder_model = Model(enc_in, [h, c])

# Build decoder model separately
dec_state_input_h = Input(shape=(64,))
dec_state_input_c = Input(shape=(64,))
dec_emb2 = model.layers[3](dec_in)
dec_out2, h2, c2 = model.layers[5](dec_emb2, initial_state=[dec_state_input_h, dec_state_input_c])
dec_out2 = model.layers[6](dec_out2)
decoder_model = Model([dec_in, dec_state_input_h, dec_state_input_c], [dec_out2, h2, c2])

# Translate
def translate(word):
    x = pad_sequences(eng_tok.texts_to_sequences([word]), maxlen=X.shape[1])
    states = encoder_model.predict(x)

    target = np.zeros((1,1))
    target[0] = hin_tok.word_index['<start>']

    result = []
    for _ in range(5):
        output_tokens, h, c = decoder_model.predict([target] + states)
        word_id = np.argmax(output_tokens[0, -1, :])
        word = hin_tok.index_word.get(word_id, '')
        if word == '<end>' or word == '':
            break
        result.append(word)
        target[0] = word_id
        states = [h, c]

    return ' '.join(result)

print(translate("hello"))
```

### **Machine Translation with Attention Machanism**

```py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Attention, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

english_sentences = [
    "I like music", "I am happy", "See you later"
]

hindi_sentences = [
    "<start> मुझे संगीत पसंद है <end>", "<start> मैं खुश हूँ <end>", "<start> बाद में मिलते हैं <end>"
]

assert len(english_sentences) == len(hindi_sentences), "Mismatch in dataset size"
print(f"Dataset size: {len(english_sentences)} sentence pairs")

def tokenize(sentences):
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(sentences)
    return tokenizer, tokenizer.texts_to_sequences(sentences)

def pad(seq, max_len):
    return pad_sequences(seq, maxlen=max_len, padding='post')

eng_tokenizer, eng_sequences = tokenize(english_sentences)
hin_tokenizer, hin_sequences = tokenize(hindi_sentences)

eng_vocab_size = len(eng_tokenizer.word_index) + 1
hin_vocab_size = len(hin_tokenizer.word_index) + 1
max_eng_len = max(len(seq) for seq in eng_sequences)
max_hin_len = max(len(seq) for seq in hin_sequences)

eng_sequences_padded = pad(eng_sequences, max_eng_len)
hin_sequences_padded = pad(hin_sequences, max_hin_len)

decoder_input_data = hin_sequences_padded[:, :-1]
decoder_target_data = hin_sequences_padded[:, 1:]

embedding_dim = 256
lstm_units = 512
batch_size = 32
epochs = 40

encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(eng_vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(hin_vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

# Attention mechanism
attention = Attention()([decoder_outputs, encoder_outputs])
decoder_concat = Concatenate(axis=-1)([decoder_outputs, attention])

decoder_dense = Dense(hin_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_concat)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#model.summary()

# Train
model.fit([eng_sequences_padded, decoder_input_data],
          np.expand_dims(decoder_target_data, -1),
          batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1)

encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

encoder_outputs_input = Input(shape=(None, lstm_units))
decoder_state_input_h = Input(shape=(lstm_units,))
decoder_state_input_c = Input(shape=(lstm_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h_inf, state_c_inf]

# Attention during inference
attention_inf = Attention()([decoder_outputs_inf, encoder_outputs_input])
decoder_concat_inf = Concatenate(axis=-1)([decoder_outputs_inf, attention_inf])
decoder_outputs_inf = decoder_dense(decoder_concat_inf)

decoder_model = Model(
    [decoder_inputs, encoder_outputs_input] + decoder_states_inputs,
    [decoder_outputs_inf] + decoder_states
)


def decode_sequence(input_seq):
    enc_output, h, c = encoder_model.predict(input_seq, verbose=0)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = hin_tokenizer.word_index['<start>']

    decoded_sentence = ''
    while True:
        output_tokens, h, c = decoder_model.predict([target_seq, enc_output] + [h, c], verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = hin_tokenizer.index_word.get(sampled_token_index, '')

        if sampled_word == '<end>' or len(decoded_sentence.split()) >= max_hin_len-1:
            break
        if sampled_word and sampled_word != '<start>':
            decoded_sentence += ' ' + sampled_word

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

    return decoded_sentence.strip()

# ============== TEST TRANSLATION ==============
test_sentence = "I like music"
test_input = pad([eng_tokenizer.texts_to_sequences([test_sentence])[0]], max_eng_len)
translated = decode_sequence(test_input)
print(f"Input: {test_sentence}")
print(f"Output: {translated}")
```
## **BERT(Bidirectional Encoder Representations from Transformers fine tuning)**
```py
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, create_optimizer

# Dataset
data = {
    'text': [
        'Excellent product!', 'Poor quality', 'Highly recommend',
        'Mediocre, not bad', 'Worst purchase ever', 'It works fine'
    ],
    'label': [1, 0, 1, 0, 0, 1]
}

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(data['text'], truncation=True, padding=True, max_length=128, return_tensors='tf')

# Model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.config.id2label = {0: "Negative", 1: "Positive"}

# Create optimizer (huggingface way)
steps_per_epoch = 1  # small data here
num_train_steps = steps_per_epoch * 3  # epochs
optimizer, _ = create_optimizer(
    init_lr=2e-5,
    num_train_steps=num_train_steps,
    num_warmup_steps=0
)

# Compile Model
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=1)
# Remove reduce_lr callback completely
history = model.fit(
    {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']},
    tf.convert_to_tensor(data['label']),
    epochs=3
)


# Save Model
model.save_pretrained("my_bert_model")
tokenizer.save_pretrained("my_bert_model")

# Inference
def predict(text):
    inputs = tokenizer(text, return_tensors='tf')
    logits = model(inputs).logits
    probabilities = tf.nn.softmax(logits).numpy()[0]
    return {
        "label": model.config.id2label[int(tf.argmax(logits, axis=1))],
        "confidence": float(max(probabilities))
    }

# Example
print(predict("Very good"))

```

## **GAN(Generative Adversarial Network)**
```py
import tensorflow as tf
import matplotlib.pyplot as plt

# Load and preprocess MNIST data
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)

BUFFER_SIZE = 60000
BATCH_SIZE = 128
LATENT_DIM = 100
EPOCHS = 50
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Generator and Discriminator models
generator = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(LATENT_DIM,)),
    tf.keras.layers.Dense(784, activation='tanh'),
    tf.keras.layers.Reshape((28, 28))
])

discriminator = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='leaky_relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Optimizers and Loss
cross_entropy = tf.keras.losses.BinaryCrossentropy()
gen_optimizer = tf.keras.optimizers.Adam(1e-4)
disc_optimizer = tf.keras.optimizers.Adam(1e-4)

# Training function
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + cross_entropy(tf.zeros_like(fake_output), fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Train GAN and generate images
def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
        if (epoch + 1) % 10 == 0:
            generate_and_plot_images(generator)

# Generate and plot images
def generate_and_plot_images(model, n=16):
    noise = tf.random.normal([n, LATENT_DIM])
    gen_images = model(noise, training=False)
    gen_images = (gen_images + 1) / 2.0  # Rescale to [0, 1]

    plt.figure(figsize=(4, 4))
    for i in range(n):
        plt.subplot(4, 4, i+1)
        plt.imshow(gen_images[i], cmap='gray')
        plt.axis('off')
        print("---")
    plt.tight_layout()
    plt.show()

# Start training
train(dataset, EPOCHS)

```
