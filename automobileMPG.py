import sklearn
import pandas
from sklearn.neighbors import KNeighborsRegressor
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import csv



names = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "modelyear", "origin", "carname"]
data = pandas.read_csv(r"C:\Users\Rochak\Desktop\Machine Learning Exercise\auto-mpg.data", delimiter ='\s\s+|\t' ,engine='python', names=names,  na_values=["?"], skipinitialspace=True, skip_blank_lines=True, header=None).dropna()
#print(data)


# =============================================================================
# #print(data.isna().any())
# #rint(data[data.isna().any(axis=1)] )
# 
# data.horsepower = data.horsepower.fillna("unknown")
# #print(data.isna().any())
# 
# 
columns_with_object = data.select_dtypes(include=["object"]).columns
print("here" ,columns_with_object)
# 
# for column in columns_with_object:    
#     print(data[column].value_counts())
# 
# =============================================================================

#Encoding
for column in columns_with_object:
    data[column] = data[column].astype('category')
    temp = column+"_cat"
    data[temp] = data[column].cat.codes
  
    


columns_with_object = data.select_dtypes(include=["object"]).columns
print(columns_with_object)    


 
data = data.drop(columns=["carname"])
      


y = data.iloc[0:, 0]
x = data.iloc[0:, 1:]
x,y = shuffle(x,y, random_state = 0)
xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = 0.3, random_state = 0)
classifier = KNeighborsRegressor(n_neighbors = 3)


classifier.fit(xTrain, yTrain)
predictions = classifier.predict(xTest)
#print(predictions)

print(classifier.score(xTest, yTest))
#score(self, x, y, sample_weight=None)[source]
# =============================================================================
# 
[print(pred, real,'\n') for pred, real in zip(predictions, yTest)]
# 
# =============================================================================
