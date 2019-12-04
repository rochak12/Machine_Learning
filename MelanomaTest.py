#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = "C:/Users/Holt/Documents/MRMFinal/Faces"
CATEGORIES = ["male", "female"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category) # path to dir
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        plt.imshow(img_array)
        plt.show()
        break
    break


# In[4]:


print(img_array.shape)


# In[5]:


IMG_SIZE = 50

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap = 'gray')
plt.show()


# In[12]:


training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # path to dir
        class_num = CATEGORIES.index(category)
        count = 0
        for img in os.listdir(path): # To Do: take only some males
            count += 1
            if (count < 2967):
                try:
                    img_array = cv2.imread(os.path.join(path,img))
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    training_data.append([new_array, class_num])
                except Exception as e:
                    pass


create_training_data()


# In[13]:


print(len(training_data))


# In[14]:


import random

random.shuffle(training_data)


# In[15]:


for sample in training_data[:10]:
    print(sample[1])


# In[16]:


X = []
y = []


# In[17]:


for features, label in training_data:
    X.append(features)
    y.append(label)

np.shape(X)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)


# In[18]:


import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# In[19]:


pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)


# In[20]:


import numpy as np

np.shape(X)


# In[21]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

X = X/255.0

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
             optimizer="adam",
             metrics=["accuracy"])

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.1)


# In[ ]:




