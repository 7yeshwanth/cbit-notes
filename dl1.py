
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
