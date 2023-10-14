'''
reads data from '.npy' files,
prepares it for classification,
builds and trains a neural network model to classify emotions,
and then saves the trained model and the labels associated with emotions.
The model aims to predict the emotion category based on the input data,
'''


import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical  # Add this import

from keras.layers import Input, Dense
from keras.models import Model

is_init = False
size = -1

label = []
dictionary = {}
c = 0

#It extracts and concatenates the data into X and the labels into y.
# Labels are also mapped to integers and stored in the dictionary.
for i in os.listdir():
    if i.split(".")[-1] == "npy" and not (i.split(".")[0] == "labels"):
        if not (is_init):
            is_init = True
            X = np.load(i)
            size = X.shape[0]
            y = np.array([i.split('.')[0]] * size).reshape(-1, 1)
        else:
            X = np.concatenate((X, np.load(i)))
            y = np.concatenate((y, np.array([i.split('.')[0]] * size).reshape(-1, 1)))

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c
        c = c + 1

for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

'''
Shuffle the data and labels for better training by randomizing the order.
It uses np.random.shuffle to shuffle the indices.
Convert emotion labels to one-hot encoded format
using to_categorical from tensorflow.keras.utils.
Create new arrays X_new and y_new to hold the shuffled data and labels.
'''

y = to_categorical(y)

X_new = X.copy()
y_new = y.copy()
counter = 0

cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)


'''
Define the architecture of the neural network model:
Input layer with the shape of the input data.
Two hidden layers with 512 and 256 neurons, both using ReLU activation functions.
Output layer with the number of neurons equal to the number of emotion categories 
and using a softmax activation function
'''
for i in cnt:
    X_new[counter] = X[i]
    y_new[counter] = y[i]
    counter = counter + 1

ip = Input(shape=(X.shape[1]))

m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)

op = Dense(y.shape[1], activation="softmax")(m)

model = Model(inputs=ip, outputs=op)

#Compile the model with the "rmsprop" optimizer,
#categorical cross-entropy loss, and accuracy as a metric.
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

model.fit(X, y, epochs=50) #Train the model using the prepared data and labels with 50 epochs.

model.save("detectormodel.h5")
np.save("emotionlabels.npy", np.array(label))