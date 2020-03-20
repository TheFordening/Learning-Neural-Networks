#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.layers import Input, Dense, Reshape,Flatten
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import numpy as np

data_1 = keras.datasets.mnist
(x_train , _),(x_test, _) = data_1.load_data()
x_train = x_train/255

input_noise = Input(shape=(1,100))
x_1 = Dense(512,activation = 'relu')(input_noise)
x_2 = Dense(1024,activation = 'relu')(x_1)
x_3 = Dense(784,activation = 'sigmoid')(x_2)
#Reshape the output to match the input to the next network
output = Reshape((28,28))(x_3)

Generator = Model(input_noise,output)

Generator.summary()

plot_model(Generator, to_file='model.png')


# In[2]:


def random_noise(output_size = 100,batch_size = 10000):
    return np.random.rand(1,100)


# In[3]:


input_image = Input(shape = (28,28))
y = Flatten()(input_image)
y = Dense(16,activation = 'relu')(y)
y = Dense(8,activation = 'relu')(y)
output_decision = Dense(1,activation = 'sigmoid')(y)

Discriminator = Model(input_image,output_decision)

sgd = keras.optimizers.SGD(lr=0.001)
Discriminator.compile(loss = 'binary_crossentropy',optimizer = sgd)

print(Discriminator.summary())


# In[5]:


def generate_dataset():
    dataset = [[x_train[i].reshape((28,28)),1] for i in range(10000)] 
    for n in range(10000):
        x_train_fake = Generator(random_noise())
        x_train_fake = np.array(x_train_fake).reshape((28,28))
        dataset.append([x_train_fake,0])
    return dataset


# In[6]:


def parse_data():
    dataset = generate_dataset()
    np.random.shuffle(dataset)
    dataset_images = [i[0] for i in dataset]
    dataset_images = np.array(dataset_images)
    dataset_labels = [i[1] for i in dataset]
    dataset_labels = np.array(dataset_labels)
    return(dataset_images,dataset_labels)

dataset_images,dataset_labels = parse_data()


# In[7]:


print(np.shape(dataset_labels))


# In[8]:


Discriminator.fit(dataset_images,dataset_labels,epochs = 100)


# In[9]:


Discriminator.predict(Generator(np.random.rand(1,1,100)))


# In[10]:


data_in = Input(shape = (1,100))
out = Discriminator(Generator(data_in))
Discriminator.trainable = False
Combined = Model(data_in,out)
Combined.compile(loss = 'binary_crossentropy',optimizer = 'adadelta')


# In[11]:


def generate_data_combined():
    noise_arr = []
    one_arr = []
    for i in range(10000):
        train_fake = random_noise()
        train_fake = np.array(train_fake)
        noise_arr.append(train_fake)
        one_arr.append(1)
    noise_arr = np.array(noise_arr)
    one_arr = np.array(one_arr)
    return noise_arr,one_arr

noise_arr,one_arr = generate_data_combined()


# In[12]:


Combined.predict(np.random.rand(1,1,100))


# In[38]:


for i in range(20):
    Discriminator.trainable = False
    Combined.fit(noise_arr,one_arr,epochs = 20)
    dataset_images,dataset_labels = parse_data()
    Discriminator.trainable = True
    Discriminator.fit(dataset_images,dataset_labels,epochs = 20)
    noise_arr,one_arr = generate_data_combined()
    
Combined.save("GANN.h5")


# In[39]:


finallys = Generator.predict(noise_arr)


# In[44]:


print(Discriminator(finallys[10].reshape(1,28,28)))
plt.imshow(np.dot(finallys[100],255),cmap = 'gray')
plt.show()


# In[ ]:




