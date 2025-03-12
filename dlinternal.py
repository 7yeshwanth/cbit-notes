
# MLPCls
import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# Similar to MNIST but...instead of 28x28 it is 8x8 and is of 1797 images rather than 70k
df = load_digits()
x = df.data
y = df.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

model=MLPClassifier(hidden_layer_sizes=(200,200),
                    activation='relu',
                    solver='adam',
                    batch_size=32,
                    max_iter=200,
                    learning_rate_init=0.01,
                    random_state=51)


model.fit(x_train,y_train)


y_pred_training = model.predict(x_train)
training_accuracy = accuracy_score(y_train, y_pred_training)



# Opt
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense
from tensorflow.keras.datasets import mnist
(X_train,y_train),(X_test,y_test)=mnist.load_data()

#MIN-MAX scaling concept:
#255/255 = 1, 0/255 = 0....120/255=0.623......all will be b/w 0 and 1
X_train = X_train/255
X_test = X_test/255

#a dictionary of optimizers
optimizers={
    'sgd':tf.keras.optimizers.SGD(learning_rate=0.1),
    'Momentum':tf.keras.optimizers.SGD(learning_rate=0.1,momentum=0.9),
    'Adam':tf.keras.optimizers.Adam(learning_rate=0.1)
}

results={}

for opt_name,optimizer in optimizers.items():
  print("Training the model with optimizer : ",opt_name)
  model=Sequential([
      Flatten(input_shape=(28,28)),
      Dense(128,activation='sigmoid'),
      Dense(64,activation='sigmoid'),
      Dense(10,activation='softmax')
  ])
  model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

  start_time = time.time()
  history=model.fit(X_train,y_train,epochs=10,batch_size=64,validation_split=0.2,verbose=0)
  end_time=time.time()

  train_loss,training_accuracy=model.evaluate(X_train,y_train,verbose=0)
  test_loss,test_accuracy=model.evaluate(X_test,y_test,verbose=0)

  print("Training Loss : ", train_loss)
  print("Training Accuracy : ",training_accuracy)
  print("Testing Loss : ", test_loss)
  print("Testing Accuracy : ", test_accuracy)
  print("Training Time ", end_time-start_time)

  results[opt_name]={
      "accuracy":test_accuracy,
      "training_time":end_time-start_time
  }
  print(results,"\n")

optimizers=list(results.keys())
accuracies=[results[opt_name]['accuracy'] for opt_name in optimizers ]
training_times=[results[opt_name]['training_time'] for opt_name in optimizers]


#Training Time comparision
plt.bar(optimizers, training_times, color=['blue', 'green', 'orange'])
plt.title("Training Time Comparison")
plt.xlabel("Optimizers")
plt.ylabel("Time (seconds)")


# Accuracy comparision
plt.bar(optimizers, accuracies, color=['blue', 'green', 'orange'])
plt.title("Accuracy Comparison")
plt.xlabel("Optimizers")
plt.ylabel("Accuracy")


# regularization

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize

# Convert labels to One-Hot Encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Use a smaller training dataset to exaggerate overfitting
x_train = x_train[:500]  # Only 5,000 samples instead of 60,000
y_train = y_train[:500]

# Define a deeper MLP model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(1024, activation='relu'),  # More neurons
    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=32, verbose=1)

# Evaluate Model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
best_epoch = history.history['val_loss'].index(min(history.history['val_loss'])) + 1

# Output Results
print("\nBaseline Model Performance:")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Best Epoch: {best_epoch}")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {min(history.history['val_loss']):.4f}")

from tensorflow.keras.regularizers import l2


model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(1024, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1024, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=32, verbose=1)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
best_epoch = history.history['val_loss'].index(min(history.history['val_loss'])) + 1

print("\nL2 Regularization Performance:")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Best Epoch: {best_epoch}")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {min(history.history['val_loss']):.4f}")

from tensorflow.keras.layers import Dropout

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=32, verbose=1)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
best_epoch = history.history['val_loss'].index(min(history.history['val_loss'])) + 1

print("\nDropout Regularization Performance:")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Best Epoch: {best_epoch}")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {min(history.history['val_loss']):.4f}")

from tensorflow.keras.layers import BatchNormalization

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=32, verbose=1)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
best_epoch = history.history['val_loss'].index(min(history.history['val_loss'])) + 1

print("\nBatch Normalization Performance:")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Best Epoch: {best_epoch}")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {min(history.history['val_loss']):.4f}")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize

# Convert labels to One-Hot Encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Use a smaller training dataset to exaggerate overfitting
x_train = x_train[:500]  # Only 500 samples instead of 60,000
y_train = y_train[:500]

# Define an overfitting MLP model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(1024, activation='relu'),
    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Apply Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train Model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                    epochs=20, batch_size=32, verbose=1, callbacks=[early_stopping])

# Evaluate Model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
best_epoch = history.history['val_loss'].index(min(history.history['val_loss'])) + 1

# Output Results
print("\nOverfitting Model with Early Stopping Performance:")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Best Epoch: {best_epoch}")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {min(history.history['val_loss']):.4f}")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

# Load dataset
(x, y), (x_test, y_test) = mnist.load_data()

# Normalize pixel values (0-255 → 0-1)
x = x / 255.0
x_test = x_test / 255.0

# Convert labels to one-hot encoding
y = to_categorical(y, 10)
y_test = to_categorical(y_test, 10)

# Split into Training (90%) and Test (10%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

# Define model with Gaussian Noise
model = Sequential([
    Flatten(input_shape=(28, 28)),
    GaussianNoise(0.1),  # Adds random noise to input for regularization
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model (No Early Stopping)
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=32, verbose=1)

# Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

# Find Best Epoch (Minimum Validation Loss)
best_epoch = history.history['val_loss'].index(min(history.history['val_loss'])) + 1

# Print performance metrics
print("\nNoise Injection Performance:")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Best Epoch: {best_epoch}")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {min(history.history['val_loss']):.4f}")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

# Load dataset
(x, y), (x_test, y_test) = mnist.load_data()

# Normalize pixel values (0-255 → 0-1)
x = x / 255.0
x_test = x_test / 255.0

# Convert labels to one-hot encoding
y = to_categorical(y, 10)
y_test = to_categorical(y_test, 10)

# Split into Training (90%) and Test (10%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

# Define model with Gaussian Noise in the Output Layer
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10),  # Linear activation before noise
    GaussianNoise(0.1),  # Injects noise into the output layer
    tf.keras.layers.Activation('softmax')  # Apply softmax after noise
])

# Compile model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=32, verbose=1)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

# Find Best Epoch (Minimum Validation Loss)
best_epoch = history.history['val_loss'].index(min(history.history['val_loss'])) + 1

# Print Performance Metrics
print("\nNoise Injection in Output Layer Performance:")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Best Epoch: {best_epoch}")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {min(history.history['val_loss']):.4f}")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

# Load dataset
(x, y), (x_test, y_test) = mnist.load_data()

# Normalize pixel values (0-255 → 0-1)
x = x / 255.0
x_test = x_test / 255.0

# Convert labels to one-hot encoding
y = to_categorical(y, 10)
y_test = to_categorical(y_test, 10)

# Split into Training (90%) and Test (10%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

# Define model with Gaussian Noise in Hidden Layers
model = Sequential([
    Flatten(input_shape=(28, 28)),
    GaussianNoise(0.1),  # Injects noise into input layer
    Dense(512, activation='relu'),
    GaussianNoise(0.05),  # Injects noise into first hidden layer
    Dense(256, activation='relu'),
    GaussianNoise(0.05),  # Injects noise into second hidden layer
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=32, verbose=1)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

# Find Best Epoch (Minimum Validation Loss)
best_epoch = history.history['val_loss'].index(min(history.history['val_loss'])) + 1

# Print Performance Metrics
print("\nNoise Injection in Hidden Layers Performance:")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Best Epoch: {best_epoch}")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {min(history.history['val_loss']):.4f}")
