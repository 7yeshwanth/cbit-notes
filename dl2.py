
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
