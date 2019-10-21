import pandas


from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report


names = ["SL", "SW" , "PL", "PW", "Class"];
data = pandas.read_csv(r"C:\Users\Rochak\Desktop\Machine Learning Exercise\iris.data", names = names)

x = data.iloc[0:, :-1]
y = data.iloc[0:, -1]
print(data.columns)
Class = (np.unique(data.Class))

xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = 0.3, random_state = 0)

classifier = DecisionTreeClassifier(criterion = "entropy", random_state= 100, max_depth = 3, min_samples_leaf = 3)
classifier.fit(xTrain, yTrain)
#tree.plot_tree(classifier.fit(xTrain, yTrain)) 
yPred = classifier.predict(xTest)

# =============================================================================
print ("Accuracy : ", accuracy_score(yTest,yPred)*100) 
# print("Confusion Matrix: ",  confusion_matrix(yTest, yPred))  
# print("Report : ", classification_report(yTest, yPred)) 
# =============================================================================


from sklearn.externals.six import StringIO  
import pydot
from IPython.display import Image
dot_data = StringIO()
tree.export_graphviz(classifier, out_file=dot_data, 
                     feature_names=data.columns[:-1] ,
                     class_names= Class) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 

graph[0].write_pdf("iris.pdf") 
#Image(graph[0].create_png()) 
