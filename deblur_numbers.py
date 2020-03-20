import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Input,Conv2D,MaxPooling2D,Dropout,UpSampling2D
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import cv2

data = keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = data.load_data()

def blur(img):
    x = []
    for i in x_train:
        x.append(cv2.GaussianBlur(i,(5,5),0))
    return np.array(x)

x_train_blur = blur(x_train)

x_train_blur = x_train_blur.reshape((len(x_train_blur),28,28,1))
x_train_blur = tf.cast(x_train_blur, tf.float32)
x_train = x_train.reshape((len(x_train),28,28,1))
x_train = tf.cast(x_train, tf.float32)/255
x_test  = x_test.reshape((len(x_test),28,28,1))
x_test = tf.cast(x_test, tf.float32)/255

#encoded
input_img = Input(shape = (28,28,1))
layer1  = Conv2D(64,(3,3,),activation  = 'relu',padding = 'same')(input_img)
encoded  = MaxPooling2D((2,2),padding = 'same')(layer1)
#decoded
layer2  =Conv2D(64,(3,3,),activation  = 'relu',padding = 'same')(encoded)
layer2 = UpSampling2D((2,2))(layer2)
decoded = Conv2D(1,(3,3),activation='sigmoid',padding = 'same')(layer2)

classify = Model(input_img,decoded)

classify.compile(loss = 'binary_crossentropy', optimizer  = 'sgd')

classify.fit(x_train_blur,x_train,epochs = 2)

x = classify.predict(x_train_blur)

for i in range(10):
    plt.imshow(x[i].reshape((28,28)))
    plt.show()
