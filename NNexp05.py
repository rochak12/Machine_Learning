import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

names_stu = ["school", "sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","guardian",
             "traveltime","studytime","failures","schoolsup","famsup","paid","activities","nursery","higher","internet",
             "romantic","famrel","freetime","goout","Dalc","Walc","health","absences","G1","G2","G3"]
#2,4,5,6,9,10,11,12,16-23,31,32
data_stu = pd.read_csv(r"C:\Users\Rochak\Desktop\Machine Learning Exercise\student\student-mat.csv", header=None, skipinitialspace=True,
                   names=names_stu, na_values=["?"], sep=';')

#deleting given headers
data_stu = data_stu.iloc[1:]

#to make sure they are the same
data_stu["school"] = data_stu.school.map({"GP": 0, "MS": 1})
data_stu["sex"] = data_stu.sex.map({"M": 0, "F": 1})
data_stu["address"] = data_stu.address.map({"R": 0, "U": 1})
data_stu["famsize"] = data_stu.famsize.map({"LE3": 0, "GT3": 1})
data_stu["Pstatus"] = data_stu.Pstatus.map({"A": 0, "T": 1})

data_stu["Mjob"] = data_stu.Mjob.map({"teacher": 0, "health": 1, "services": 2, "at_home": 3, "other": 4})
data_stu["Fjob"] = data_stu.Fjob.map({"teacher": 0, "health": 1, "services": 2, "at_home": 3, "other": 4})
data_stu["reason"] = data_stu.reason.map({"home": 0, "reputation": 1, "course": 2, "other": 3})
data_stu["guardian"] = data_stu.guardian.map({"mother": 0, "father": 1, "other": 2})


data_stu["schoolsup"] = data_stu.schoolsup.map({"no": 0, "yes": 1})
data_stu["famsup"] = data_stu.famsup.map({"no": 0, "yes": 1})
data_stu["paid"] = data_stu.paid.map({"no": 0, "yes": 1})
data_stu["activities"] = data_stu.activities.map({"no": 0, "yes": 1})
data_stu["nursery"] = data_stu.nursery.map({"no": 0, "yes": 1})
data_stu["higher"] = data_stu.higher.map({"no": 0, "yes": 1})
data_stu["internet"] = data_stu.internet.map({"no": 0, "yes": 1})
data_stu["romantic"] = data_stu.romantic.map({"no": 0, "yes": 1})

#data_stu = data_stu.drop(columns=["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian",
#                                "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"])

# pd.set_option('display.max_columns', 999)
#was getting a type error so changed all the objects to floats
data_stu["age"] = data_stu.age.astype(float)
data_stu["Medu"] = data_stu.Medu.astype(float)
data_stu["Fedu"] = data_stu.Fedu.astype(float)
data_stu["traveltime"] = data_stu.traveltime.astype(float)
data_stu["studytime"] = data_stu.studytime.astype(float)
data_stu["failures"] = data_stu.failures.astype(float)
data_stu["famrel"] = data_stu.famrel.astype(float)
data_stu["freetime"] = data_stu.freetime.astype(float)
data_stu["goout"] = data_stu.goout.astype(float)
data_stu["Dalc"] = data_stu.Dalc.astype(float)
data_stu["Walc"] = data_stu.Walc.astype(float)
data_stu["health"] = data_stu.health.astype(float)
data_stu["absences"] = data_stu.absences.astype(float)
data_stu["G1"] = data_stu.G1.astype(float)
data_stu["G2"] = data_stu.G2.astype(float)
data_stu["G3"] = data_stu.G3.astype(float)
#print(data_stu.dtypes)
# print(data_stu)

#calculating final grade
X = data_stu.drop(columns=["G3"]).values
y = data_stu["G3"].values.flatten()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()

model.add(Dense(45, input_shape=(32,), activation='relu', name='fc1'))
model.add(Dense(100, activation='relu', name='fc2'))
model.add(Dense(100, activation='relu', name='fc3'))
model.add(Dense(100, activation='relu', name='fc4'))
model.add(Dense(21, activation='softmax', name='output'))

optimizer = Adam(lr=.01)
model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print('Neural Network Model Summary: ')
print(model.summary())

model.fit(X_train, y_train, verbose=2, batch_size=10, epochs=100)

results = model.evaluate(X_test, y_test)

#the best I got was 44%
#average in the low 30's
print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))
