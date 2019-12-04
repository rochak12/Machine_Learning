from sklearn.preprocessing import OneHotEncoder
import os
import numpy as np
from sklearn.metrics import accuracy_score


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense
from keras.optimizers import Adam


names = ["Sepal Length", "Sepal Width" , "Petal length", "Petal Width", "Species"]
data = pd.read_csv(r"C:\Users\Rochak\Desktop\Machine Learning Exercise\iris.data", header=None, skipinitialspace=True,
                   names=names, na_values=["?"], sep=',')


columns_with_object = data.select_dtypes(include=["object"]).columns
for column in columns_with_object:
    data[column] = data[column].astype('category')
    temp = column+"_cat"
    data[temp] = data[column].cat.codes       
data = data.drop(columns = ["Species"])


x = preprocessing.scale(data.iloc[:, :-1])
y = to_categorical(data.iloc[0:, -1])
xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = 0.3)

print(y)

#model
model = Sequential()
model.add(Dense(10, input_dim=4, activation= 'relu', name = 'fc1'))
model.add(Dense(10, activation='relu', name='fc2'))
model.add(Dense(3, activation ='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
              
model.fit(xTrain, yTrain, validation_data = (xTest, yTest), epochs=1000, batch_size = 100)


results = model.evaluate(xTest, yTest)
print(results)

#accuracy 98%