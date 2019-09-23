


#Using sklearn library
from sklearn import datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target

#same thing done with train_test_split library
from sklearn.utils import shuffle
x,y = shuffle(x, y , random_state = 0)


from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3, random_state = 0)

from sklearn.naive_bayes import BernoulliNB
import sklearn
classifier = sklearn.naive_bayes.GaussianNB()
classifier.fit(xTrain, yTrain)

target_predicated = classifier.predict(xTest)
#print(yTest, target_predicated)


from sklearn.metrics import accuracy_score
print(accuracy_score(yTest, target_predicated) * 100)




class HardCodedClassifier:
    
    def fit(self, xtrain, ytrain):
        pass
    
    def predict(self, xtest):
        ytest = [0]* len(xtest)
        return ytest
        
    
class1  = HardCodedClassifier()
class1.fit(xTrain, yTrain)
aaa = class1.predict(yTest)
print(accuracy_score(yTest, aaa) * 100)
print(yTest, aaa)




#using pandas library
#import pandas
#raw_data = open(r"C:\Users\Rochak\Desktop\Machine Learning Exercise\iris.data")
#data = pandas.read_csv(r"C:\Users\Rochak\Desktop\Machine Learning Exercise\iris.data")
#data = numpy.loadtxt(raw_data, delimiter = ",") 
#x = data.iloc[:, 0:4 ]
#y = data.iloc[:, -1] 


'''
jhsvchjavbsck
'''







