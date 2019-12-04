import pandas
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# =============================================================================
# iris = datasets.load_iris()
# x = iris.data
# y = iris.target
# 
# #same thing done with train_test_split library
# x,y = shuffle(x, y , random_state = 0)
# 
# 
# xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3, random_state = 0)
# 
# classifier = KNeighborsClassifier(n_neighbors=7, metric = 'minkowski' , p=2)
# classifier.fit(xTrain, yTrain)
# 
# target_predicated = classifier.predict(xTest)
# #print(yTest, target_predicated)
# 
# 
# print(accuracy_score(yTest, target_predicated) * 100)
# 
# =============================================================================
############################################################################################




columns = ['no-recurrence-events',  '30-39',  'premeno', '30-34',
                                         '0-2', 'no', '3',  'left', 'left_low', 'no.1']
data = pandas.read_csv(r"C:\Users\Rochak\Desktop\Machine Learning Exercise\breast-cancer.data", names = columns)
#print(data)

print(data.head())
#[print(column) for column in data.columns]

columns_with_object = data.select_dtypes(include=["object"]).columns
for column in columns_with_object:
    data[column] = data[column].astype('category')
    temp = column+"_cat"
    data[temp] = data[column].cat.codes
    
for column in columns_with_object:
    data = data.drop(columns=[column])


x = data.iloc[0:, :-1]
y = data.iloc[0:, -1]
#be careful while using zip, sometimes it zips to the heading of x not x
#[print(xx, yy) for xx, yy in zip(numpy.array(x),y)]
x,y = shuffle(x,y, random_state = 72)
xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = 0.2, random_state = 0)
classifier = KNeighborsClassifier(n_neighbors=8, metric = 'minkowski' , p=2)
classifier.fit(xTrain, yTrain)
predictions = classifier.predict(xTest)
print(accuracy_score(yTest, predictions) * 100)

############################################3


# =============================================================================
# names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "classvalue"]
# data = pandas.read_csv(r"C:\Users\Rochak\Desktop\Machine Learning Exercise\car.data", names=names, na_values=["?"], skipinitialspace = True, header=None)
# 
# 
# 
# columns_with_object = data.select_dtypes(include=["object"]).columns
# #Encoding
# for column in columns_with_object:
#     data[column] = data[column].astype('category')
#     temp = column+"_cat"
#     data[temp] = data[column].cat.codes    
# columns_with_object = data.select_dtypes(include=["object"]).columns
# print(columns_with_object)    
# #print(data)
# 
#    
# data = data.drop(columns=["buying", "maint", "doors", "persons", "lug_boot", "safety", "classvalue"])
#   
# 
# 
#     
# x = data.iloc[0:, :-1]
# y = data.iloc[0:, -1]
# x,y = shuffle(x,y, random_state = 0)
# xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = 0.3, random_state = 0)
# classifier = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski' , p=2)
# classifier.fit(xTrain, yTrain)
# predictions = classifier.predict(xTest)
# print(accuracy_score(yTest, predictions) * 100)
# =============================================================================




