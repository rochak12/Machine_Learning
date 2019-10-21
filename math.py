import sklearn
import pandas
from sklearn.neighbors import KNeighborsRegressor
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import csv



data = pandas.read_csv(r"C:\Users\Rochak\Desktop\Machine Learning Exercise\student\student-mat.csv", delimiter =';' ,engine='python',  na_values=["?"], skipinitialspace=True, skip_blank_lines=True, header=0)
#print(data.columns)




#print(data.isna().any())
#print(data[data.isna().any(axis=1)] )

columns_with_object = data.select_dtypes(include=["object"]).columns
print("here" ,columns_with_object)

 
#for column in columns_with_object:    
#  print(data[column].value_counts())



#Encoding
for column in columns_with_object:
    data[column] = data[column].astype('category')
    temp = column+"_cat"
    data[temp] = data[column].cat.codes
  
    


for column in columns_with_object:
    data = data.drop(columns=[column])
      

x = data.drop(columns=["G3"])
y = data["G3"]

x,y = shuffle(x,y, random_state = 0)
xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = 0.3, random_state = 0)
classifier = KNeighborsRegressor(n_neighbors = 3)

classifier.fit(xTrain, yTrain)
predictions = classifier.predict(xTest)
print(predictions)

print(classifier.score(xTest, yTest))
[print(pred,'\t\t', real,'\n') for pred, real in zip(predictions, yTest)]
