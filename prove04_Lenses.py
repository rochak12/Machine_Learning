import pandas
import random
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report


names = ["No", "Age" , "Spect_Pres", "Astigmatic", "TPR", "Class"];
data = pandas.read_csv(r"C:\Users\Rochak\Desktop\Machine Learning Exercise\lenses.data", delimiter = " ", skipinitialspace=True, names = names)

x = data.iloc[0:, 1:-1]
y = data.iloc[0:, -1]
new_y = []
for yi in y:
    if yi == 1:
        new_y.append("hard contacy lens")
    if yi == 2:
        new_y.append("soft contact lens")
    if yi == 3:
        new_y.append("no contact lens")
y = new_y
data.iloc[:, -1] = y
xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = 0.3, random_state = 0)

classifier = DecisionTreeClassifier(criterion = "entropy", 
                                    random_state= 100,
                                    max_depth = 5,
                                    min_samples_leaf = 4)
classifier.fit(xTrain, yTrain)
yPred = classifier.predict(xTest)

print(data.head)
#print("Confusion Matrix: ",  confusion_matrix(yTest, yPred))  
print ("Accuracy : ", accuracy_score(yTest,yPred)*100) 
#print("Report : ", classification_report(yTest, yPred)) 



from sklearn import tree
from sklearn.externals.six import StringIO  
import pydot
from IPython.display import Image
dot_data = StringIO()
Class = (np.unique(data.Class))
tree.export_graphviz(classifier, out_file=dot_data, 
                     feature_names=data.columns[1:-1] ,
                     class_names= Class) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 

graph[0].write_pdf("lenses.pdf") 
#Image(graph[0].create_png()) 
