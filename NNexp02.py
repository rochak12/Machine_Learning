import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.utils import to_categorical   #for hotencoding
from keras.layers import Dense
from keras.optimizers import Adam


data = pd.read_csv(r"C:\Users\Rochak\Desktop\Machine Learning Exercise\student\student-mat.csv", delimiter =';' ,engine='python',  na_values=["?"], skipinitialspace=True, skip_blank_lines=True, header=0)

columns_with_object = data.select_dtypes(include=["object"]).columns
print("here" ,columns_with_object)

# =============================================================================
# for column in columns_with_object:    
#   print(data[column].value_counts())
# =============================================================================


#Encoding
for column in columns_with_object:
    data[column] = data[column].astype('category')
    temp = column+"_cat"
    data[temp] = data[column].cat.codes
  
for column in columns_with_object:
    data = data.drop(columns=[column])
    
# print(data)
# print(data.dtypes)
# print("Median age" , data.age.median())
# print(data.native_country.value_counts())
#print(len(data.columns))     
# [print(column) for column in data.columns]
# print(data.head())
# print(data.isna().any())
    
     

x = data.drop(columns=["G3"])
y = (data["G3"])
x,y = shuffle(x,y, random_state = 0)
xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = 0.3, random_state = 0)


print((data["G3"].value_counts()))  
print(y.shape)
print(y)


model = Sequential()
model.add(Dense(45, input_dim = 32, activation='relu', name='fc1'))
model.add(Dense(100, activation='relu', name='fc2'))
model.add(Dense(100, activation='relu', name='fc3'))
model.add(Dense(100, activation='relu', name='fc4'))
model.add(Dense(1, activation='linear', name='output'))

optimizer = Adam(lr=.01)
model.compile(optimizer, loss='mean_absolute_error', metrics=['accuracy'])

print('Neural Network Model Summary: ')
print(model.summary())

model.fit(xTrain, yTrain, validation_data = (xTest, yTest) ,validation_split = 0.2, batch_size=100, epochs=1000)

#test the model
predictions = model.predict(xTest)

#Accuracy calculator
range = 0
notrange = 0
for pred, yTe in zip(predictions, yTest):
    print(pred , yTe)
    if ((pred - yTe <= 2 and pred - yTe >= 0) or (yTe - pred <= 2 and yTe - pred >= 0)):
        range += 1
    else:
        notrange += 1
print(range/(notrange+range) *100 )        


