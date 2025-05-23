# Deep Learning Lab Exteral Sets

> ***🌟<u>All the best</u> 🚀***

## ***MLP Sklearn***

```py
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time

# Load and preprocess data
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)

configs = [
    {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam'},
    {'hidden_layer_sizes': (100, 50), 'activation': 'relu', 'solver': 'adam'},
    {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'sgd'},
    {'hidden_layer_sizes': (200, 100), 'activation': 'relu', 'solver': 'adam'},
    {'hidden_layer_sizes': (50,), 'activation': 'relu', 'solver': 'adam'}
]

for idx, config in enumerate(configs, 1):
    print(f"Training Configuration {idx}...")
    start_time = time.time()
    
    mlp = MLPClassifier(**config, max_iter=10, random_state=42)
    mlp.fit(X_train, y_train)
    
    train_pred = mlp.predict(X_train)
    test_pred = mlp.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    duration = time.time() - start_time

    print(f"Config {idx}:")
    print(f"  Layers: {config['hidden_layer_sizes']}, Activation: {config['activation']}, Solver: {config['solver']}")
    print(f"  Training Accuracy: {train_acc * 100:.2f}%")
    print(f"  Testing Accuracy: {test_acc * 100:.2f}%")
    print(f"  Convergence Time: {duration:.2f} seconds")
    print()
```

## ***MLP tensorflow***
```py
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Configurations
configurations = [
    {"name": "Configuration 1: SGD without momentum", "optimizer": SGD(learning_rate=0.01, momentum=0.0)},
    {"name": "Configuration 2: SGD with momentum (0.9)", "optimizer": SGD(learning_rate=0.01, momentum=0.9)},
    {"name": "Configuration 3: Adam optimizer", "optimizer": Adam()},
    {"name": "Configuration 4: SGD with Nesterov momentum", "optimizer": SGD(learning_rate=0.01, momentum=0.9, nesterov=True)},
    {"name": "Configuration 5: Adam with modified learning rate (0.0005)", "optimizer": Adam(learning_rate=0.0005)},
]

# Train and evaluate each configuration
for config in configurations:
    print(f"\n=== {config['name']} ===")

    # Define model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile model with specific optimizer
    model.compile(optimizer=config["optimizer"],
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model with timing
    start_time = time.time()
    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=20,
                        batch_size=32,
                        verbose=0)
    elapsed_time = time.time() - start_time

    # Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    train_acc = history.history['accuracy'][-1]
    best_epoch = history.history['val_loss'].index(min(history.history['val_loss'])) + 1

    # Output results
    print(f"Training Accuracy : {train_acc * 100:.2f}%")
    print(f"Testing Accuracy  : {test_acc * 100:.2f}%")
    print(f"Convergence Time  : {elapsed_time:.2f} seconds")
    print(f"Best Epoch        : {best_epoch}")
    print(f"Final Val Loss    : {min(history.history['val_loss']):.4f}")
```


## ***Regularization***
```py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Define model types
models = {
    "Baseline": lambda: Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ]),

    "L2 Regularization": lambda: Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(512, activation='relu', kernel_regularizer='l2'),
        Dense(256, activation='relu', kernel_regularizer='l2'),
        Dense(128, activation='relu', kernel_regularizer='l2'),
        Dense(10, activation='softmax')
    ]),

    "Dropout": lambda: Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ]),

    "Batch Normalization": lambda: Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ]),

    "Early Stopping": lambda: Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ]),

    "Input Noise Injection": lambda: Sequential([
        Flatten(input_shape=(28, 28)),
        GaussianNoise(0.1),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ]),

    "Hidden Layer Noise Injection": lambda: Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(512, activation='relu'),
        GaussianNoise(0.05),
        Dense(256, activation='relu'),
        GaussianNoise(0.05),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ]),

    "Output Layer Noise Injection": lambda: Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(10),
        GaussianNoise(0.1),
        tf.keras.layers.Activation('softmax')
    ])
}


early_stop = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

for name, build_fn in models.items():
    print(f"Training: {name}")
    model = build_fn()
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    epochs = 50 if 'Early Stopping' in name else 20
    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=epochs,
                        batch_size=32,
                        callbacks=[early_stop] if 'Early Stopping' in name else None,
                        verbose=0)

    train_acc = history.history['accuracy'][-1] * 100
    test_acc = history.history['val_accuracy'][-1] * 100
    overfit_gap = train_acc - test_acc

    print(f"Train Acc (%): {train_acc:.2f}")
    print(f"Test Acc (%): {test_acc:.2f}")
    print(f"Overfit Gap: {overfit_gap:.2f}")
    print()

```


## ***Denoising***
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

## ***bert***
```py
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, create_optimizer

# Sample data
texts = [
    'Excellent product!',
    'Poor quality',
    'Highly recommend',
    'Mediocre, not bad',
    'Worst purchase ever',
    'It works fine'
]
labels = [1, 0, 1, 0, 0, 1]  # 1 = Positive, 0 = Negative

# Load tokenizer and tokenize texts
tk = BertTokenizer.from_pretrained('bert-base-uncased')
encodings = tk(texts, padding=True, truncation=True, max_length=128, return_tensors='tf')

# Load model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Set up optimizer
opt, _ = create_optimizer(init_lr=2e-5, num_train_steps=3, num_warmup_steps=0)

# Compile
model.compile(
    optimizer=opt,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train
model.fit(
    {'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask']},
    tf.convert_to_tensor(labels),
    epochs=3
)


# Prediction function
def predict(text):
    inps = tk(text, return_tensors='tf', padding=True, truncation=True, max_length=128)
    logits = model(inps).logits
    probs = tf.nn.softmax(logits, axis=1)
    pred = int(tf.argmax(probs, axis=1)[0])
    conf = float(probs[0][pred])
    return {"sentiment": "Positive" if pred == 1 else "Negative", "confidence": conf}

# Test
print(predict("Very good"))  # Should return: {"sentiment": "Positive", "confidence": ~0.99}
```


## ***RNN Sentiment***
```py
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense
import matplotlib.pyplot as plt

# Hyperparameters
num_words = 5000
max_len = 100
embedding_dim = 64

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
x_train, x_test = sequence.pad_sequences(x_train, maxlen=max_len), sequence.pad_sequences(x_test, maxlen=max_len)

# Define RNN models
def build_model(rnn_layer):
    model = Sequential([
        Embedding(num_words, embedding_dim, input_length=max_len),
        rnn_layer,
        Dense(1, activation='sigmoid')
    ])
    return model

models_config = [
    {'name': 'SimpleRNN', 'layer': SimpleRNN(64)},
    {'name': 'LSTM', 'layer': LSTM(64)},
    {'name': 'GRU', 'layer': GRU(64)}
]

results = []

for config in models_config:
    print(f"\n=== Training {config['name']} ===")
    
    model = build_model(config['layer'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    history = model.fit(x_train, y_train,
                        validation_split=0.2,
                        epochs=3,
                        batch_size=128,
                        verbose=0)

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Store results
    results.append({'name': config['name'], 'history': history, 'test_acc': test_acc})

    # Sample prediction
    sample_idx = [0, 1, 2]
    sample_pred = model.predict(x_test[sample_idx])
    print("Sample Predictions vs True Labels:")
    for i in sample_idx:
        print(f"Pred: {sample_pred[i][0]:.4f} | Actual: {y_test[i]}")

# Plot training performance comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for res in results:
    plt.plot(res['history'].history['acc'], label=f"{res['name']}_train")
    plt.plot(res['history'].history['val_acc'], '--', label=f"{res['name']}_val")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
for res in results:
    plt.plot(res['history'].history['loss'], label=f"{res['name']}_train")
    plt.plot(res['history'].history['val_loss'], '--', label=f"{res['name']}_val")
plt.title("Loss")
plt.legend()
plt.show()

# Summary comparison
print("\n=== Model Comparison ===")
for res in results:
    print(f"{res['name']}: Test Acc = {res['test_acc']:.4f}")
```

[Colab link](https://colab.research.google.com/drive/1Ichfj_fo9nj5AWhzDwL4nch9sa9gewoo?authuser=1#scrollTo=8bltWhMxvbg_)