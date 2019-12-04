import pandas
from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics



#Date parser is for parsing date to date object
#But if pandas recognizes dare then parse_date or to datetime is enough.
#dateparser = lambda x: pandas.datetime.strptime(x, '%b %d, %Y')
data = pandas.read_csv(r"C:\Users\Rochak\Desktop\Machine Learning Exercise\oil.csv",
                       na_values=["?"], skipinitialspace = True) 
                         #parse_dates= ["Date"], date_parser = dateparser)
data['Date'] = pandas.to_datetime(data.Date)

#print(data.isnull().values.any())
#print(data.dtypes)

# =============================================================================
# y.plot(figsize=(12,3))  
# data.hist() 
# =============================================================================



data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day
data['year'] = data['Date'].dt.year
data = data.drop(columns = ['Date'])
print(data.head())


x = data.iloc[:, 1:]
y = data.iloc[0:, 0]
xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = 0.3)
#classifier = DecisionTreeClassifier(criterion = "entropy", random_state= 100, max_depth = 5, min_samples_leaf = 6)
classifier = LinearRegression()
#classifier = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski' , p=2)


classifier.fit(xTrain, yTrain)
#tree.plot_tree(classifier.fit(xTrain, yTrain)) 
yPred = classifier.predict(xTest)



range = 0
notrange = 0
for pred, yTe in zip(yPred, yTest):
    print( pred, "and", yTe, "and")
#print(zip(yTest, xTest))
    if (pred - yTe <= 5 and pred - yTe >= 0) or (yTe - pred <= 5 and yTe - pred >= 0):
        range += 1
    else:
        notrange += 1
print(range/(notrange+range) *100 ) 
print("Accuracy: ",  metrics.r2_score(yTest,yPred))

# =============================================================================
# 
# from sklearn import tree
# from sklearn.externals.six import StringIO  
# import pydot
# from IPython.display import Image
# dot_data = StringIO()
# Class = (np.unique(data.Price))
# tree.export_graphviz(classifier, out_file=dot_data, 
#                      feature_names=data.columns[1:]) 
# graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
# 
# graph[0].write_pdf("oil.pdf") 
# 
# ==========================================================================














# =============================================================================
# 
# import pandas
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn import datasets
# from sklearn.utils import shuffle
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn import linear_model
# from sklearn import metrics
# 
# 
# 
# 
# 
# data = pandas.read_csv(r"C:\Users\Rochak\Desktop\Machine Learning Exercise\oil.csv", na_values=["?"], skipinitialspace = True)
# print(data.head)
# data['Date'] = pandas.to_datetime(data.Date)
# 
# data['month'] = data['Date'].dt.month
# data['day'] = data['Date'].dt.day
# data['year'] = data['Date'].dt.year
# data = data.drop(columns = ['Date'])
# print(data.head())
# 
# 
# 
# 
#     
# x = data.iloc[0:, 1:]
# y = data.iloc[0:, 0]
# #x,y = shuffle(x,y, random_state = 0)
# xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = 0.3, shuffle=False)
# classifier = linear_model.LinearRegression()
# classifier.fit(xTrain, yTrain)
# predictions = classifier.predict(xTest)
# 
# range = 0
# notrange = 0
# for pred, yTe in zip(predictions, yTest):
#     if (pred - yTe < 0.1 or yTe - pred < 1):
#         range += 1
#     else:
#         notrange += 1
# print(range/(notrange+range) *100 )  
# # #print(accuracy_score(yTest, predictions) * 100)
# # print(metrics.r2_score(yTest,predictions))
# 
# =============================================================================
