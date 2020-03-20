import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import os, os.path
###########################################################
############################################################
data = keras.datasets.mnist
(train_images,train_labels),(test_images,test_labels) = data.load_data()
print(np.shape(train_images))

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(16,activation = "relu"),
    keras.layers.Dense(10,activation = "softmax")
])

model.compile(optimizer="adam", loss = "sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(train_images, train_labels, epochs = 10)
test_acc, test_loss = model.evaluate(test_images,test_labels)

prediction = model.predict(test_images)

print(np.argmax(prediction[0]))

plt.imshow(cv2.GaussianBlur(test_images[0],(7,7),0))
plt.show()

m = model.get_weights()
print(m)
