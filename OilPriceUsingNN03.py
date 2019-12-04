import pandas
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.utils import to_categorical   #for hotencoding
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn import metrics


data = pandas.read_csv(r"C:\Users\Rochak\Desktop\Machine Learning Exercise\oil.csv", na_values=["?"], skipinitialspace = True)
print(data.head)

print(data.iloc[: , 1])

data['Date'] = pandas.to_datetime(data.Date)

data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day
data['year'] = data['Date'].dt.year
data = data.drop(columns = ['Date'])
print(data)

   

#Spliting data without randomization
x = data.iloc[0:, 1:]
y = data.iloc[0:, 0]
x,y = shuffle(x,y, random_state = 0)
xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = 0.3, shuffle=True)


#neural network
model = Sequential()
model.add(Dense(100, kernel_initializer = 'normal' , input_dim = 3, activation = 'relu'))
model.add(Dense(100, kernel_initializer='normal',activation='relu'))
model.add(Dense(100, kernel_initializer='normal',activation='relu'))
model.add(Dense(1, kernel_initializer='normal',activation='linear'))
model.compile(loss='mean_absolute_error', optimizer = 'adam' , metrics = ['accuracy'])
model.summary()

#Train the model
model.fit(xTrain, yTrain, epochs=100, batch_size=1000, validation_split = 0.2)

#test the model
predictions = model.predict(xTest)


#Accuracy calculator
range = 0
notrange = 0
for pred, yTe in zip(predictions, yTest):
    print( pred, "and", yTe)
#print(zip(yTest, xTest))
    if (pred - yTe <= 5 and pred - yTe >= 0) or (yTe - pred <= 5 and yTe - pred >= 0):
        range += 1
    else:
        notrange += 1
print(range/(notrange+range) *100 )        
results = model.evaluate(xTest, yTest)
print(results)
