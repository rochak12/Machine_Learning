from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense
from keras import backend as K
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from sklearn.utils import shuffle


epochs = 50
batch_size = 20
img_width, img_height = 150, 150
test_data_dir = 'C:/Users/Rochak/Desktop/Machine Learning Exercise/dogs-vs-cats/test1/test1'
train_data_dir = 'C:/Users/Rochak/Desktop/Machine Learning Exercise/dogs-vs-cats/train/train'




#print(K.image_data_format())
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)



#loading all the file names
filenames = os.listdir(train_data_dir)
labels = []
for name in filenames:
    label = name.split('.')[0]
    if label == 'dog':
        labels.append(1)
    else:
        labels.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'label': labels
})
    
print(df.shape)
df['label'] = df['label'].replace({0:'cat', 1:'dog'})
print(df)

df = shuffle(df)
df = df.iloc[1:2000, :]

#spliting filenames in training and testing
train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
print(train_df.shape, test_df.shape )
print(train_df)



#image generator
train_data = ImageDataGenerator(
    rescale=1 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_data = ImageDataGenerator(rescale=1 / 255)



train_generator = train_data.flow_from_dataframe(
    dataframe = train_df, 
    directory = train_data_dir,
    class_mode='binary',
    x_col="filename",
    y_col='label',
    batch_size = batch_size,
    target_size= (img_width, img_height))


test_generator = test_data.flow_from_dataframe(
    test_df,   train_data_dir, 
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    x_col="filename",
    y_col='label')



#model for neural network
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.summary()

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit_generator(
    train_generator,
    validation_data=test_generator,
    steps_per_epoch= 10,
    epochs= 50,
    validation_steps= 10)

#model.save_weight('first_try')




final_test_filenames = os.listdir(test_data_dir)
final_test_df = pd.DataFrame({
    'id': final_test_filenames
})
final_test_df.shape
final_test_df = final_test_df.iloc[0:100 , :]

final_test_gen = ImageDataGenerator(rescale=1./255)
final_test_generator = final_test_gen.flow_from_dataframe(
    final_test_df, 
    test_data_dir, 
    x_col='id',
    y_col=None,
    class_mode=None,
    target_size=(150,150),
    batch_size=20
)

predict = model.predict_generator(final_test_generator, steps=625)

print(predict)
