from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import pandas
from sklearn import preprocessing
from keras.utils import to_categorical

names = ["Sepal Length", "Sepal Width" , "Petal length", "Petal Width", "Species"]
iris = pandas.read_csv(r"C:\Users\Rochak\Desktop\Machine Learning Exercise\iris.data", header=None, skipinitialspace=True,
                   names=names, na_values=["?"], sep=',')

columns_with_object = iris.select_dtypes(include=["object"]).columns
for column in columns_with_object:
    iris[column] = iris[column].astype('category')
    temp = column+"_cat"
    iris[temp] = iris[column].cat.codes
    
for column in columns_with_object:
    iris = iris.drop(columns=[column])
  
    
x = preprocessing.scale(iris.iloc[:, :-1])
y = to_categorical(iris.iloc[0:, -1])
xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = 0.3)    
    

columns_with_object = iris.select_dtypes(include=["object"]).columns
print(iris)
print("bjbjcbjdbcd")


#same thing done with train_test_split library
x,y = shuffle(x, y , random_state = 0)
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3, random_state = 0)

model = Sequential()
model.add(Dense(50, kernel_initializer = 'normal' , input_dim = 4, activation = 'relu'))
model.add(Dense(50, kernel_initializer='normal',activation='relu'))
model.add(Dense(3, kernel_initializer='normal',activation='softmax'))
model.compile(loss='mean_absolute_error', optimizer = 'adam' , metrics = ['accuracy'])
model.summary()

#Train the model
#model.fit(xTrain, yTrain, epochs=500, batch_size=32, validation_split = 0.2)
model.fit(xTrain, yTrain, validation_data = (xTest, yTest), epochs=500)


target_predicated = model.predict(xTest)
#print(yTest, target_predicated)

results = model.evaluate(xTest, yTest)
print(results)






