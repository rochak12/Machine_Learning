from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier



iris = datasets.load_iris()
x = iris.data
y = iris.target

#same thing done with train_test_split library
x,y = shuffle(x, y , random_state = 0)


xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3, random_state = 0)

classifier = DecisionTreeClassifier(criterion = "entropy", random_state= 100, max_depth = 3, min_samples_leaf = 8)
classifier.fit(xTrain, yTrain)

target_predicated = classifier.predict(xTest)
#print(yTest, target_predicated)


print(accuracy_score(yTest, target_predicated) * 100)