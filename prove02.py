import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# =============================================================================
# 
# iris = datasets.load_iris()
# x = iris.data
# y = iris.target
# print(list(zip(x,y)))
# x,y = shuffle(x,y, random_state = 0)
# 
# =============================================================================



# Use pandas to import non-numeric data its easier
# =============================================================================
# import pandas
# import numpy
# data = pandas.read_csv(r"C:\Users\Rochak\Desktop\Machine Learning Exercise\iris.data")
# x = data.iloc[0:, :-1]
# y = data.iloc[0:, -1]
# #be careful while using zip, sometimes it zips to the heading of x not x
# [print(xx, yy) for xx, yy in zip(numpy.array(x),y)]
# x,y = shuffle(x,y, random_state = 0)
# xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = 0.3, random_state = 0)
# classifier = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski' , p=2)
# classifier.fit(xTrain, yTrain)
# predictions = classifier.predict(xTest)
# print(accuracy_score(yTest, predictions) * 100)
# 
# =============================================================================



import pandas
import numpy
data = pandas.read_csv(r"C:\Users\Rochak\Desktop\Machine Learning Exercise\breast-cancer.data")
#print(data)

#print(data.head())
#[print(column) for column in data.columns]

df = pandas.DataFrame(data, columns = ['no-recurrence-events',  '30-39',  'premeno', '30-34', '0-2', 'no', '3',  'left', 'left_low', 'no.1'])
print(df)
x = data.iloc[0:, :-1]
y = data.iloc[0:, -1]
#be careful while using zip, sometimes it zips to the heading of x not x
#[print(xx, yy) for xx, yy in zip(numpy.array(x),y)]
# =============================================================================
# x,y = shuffle(x,y, random_state = 0)
# 
# 
# xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = 0.3, random_state = 0)
# classifier = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski' , p=2)
# classifier.fit(xTrain, yTrain)
# predictions = classifier.predict(xTest)
# =============================================================================
#print(accuracy_score(yTest, predictions) * 100)

