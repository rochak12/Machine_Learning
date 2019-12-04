from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


# =============================================================================
# # Use pandas to import non-numeric data its easier
names = ["PL", "PW", "SL", "SW", "Class"]
data = pd.read_csv(r"C:\Users\Rochak\Desktop\Machine Learning Exercise\iris.data", names=names)
x = data.iloc[0:, :-1]
y = data.iloc[0:, -1]
# print(data.head())
# x,y = shuffle(x,y, random_state = 0)
# xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = 0.3, random_state = 0)
# 
# =============================================================================


# =============================================================================
# #This idea failed to convert into 3 different option like 001, 100, 010
# x = np.array(x)
# y = np.array(y).T
# print(y)
# y1 = []
# for i in y:
#     if i == "Iris-setosa": 
#         y1.extend([1,0,0])
#     if i == "Iris-versicolor":
#         y1.extend([0,1,0])
#     if i == "Iris-virginica":
#         y1.extend([0,0,1])
# 
# y1 = np.array(y1)
# y = y1.reshape((len(y),3))
# =============================================================================

# =============================================================================
# x = np.array([[1, 0, 0, 0],
#               [0, 1, 0, 0],
#               [0,1,1,0],
#               [1,0,0,1],
#               [1,0,0,0],
#               [1,0,0,1]])
# 
# y = np.array([[1 ,1 ,0 ,0 ,1,0]]).T
# =============================================================================

# =============================================================================
# y = np.array([[1,0,0],
#               [0,1,0],
#               [0,1,0],
#               [0,0,1],
#               [1,0,0],
#               [0,0,1]])
# =============================================================================



# =============================================================================
# y = np.array([[0],
#               [1],
#               [1],
#               [2],
#               [0],
#               [0]])
# =============================================================================

x = np.array(x)
y1 = []


'''
#One output Node 
# =============================================================================
# for i in y:
#     if i == "Iris-setosa": 
#         y1.append(0)
#     if i == "Iris-versicolor":
#          y1.append(1)
#     if i == "Iris-virginica":
#         y1.append(2)
# 
# y = (np.array(y1)).reshape((len(y),1))
# #print(x,y)
# =============================================================================
##############################################3
'''

#Output node same as number of outputs
for i in y:
    if i == "Iris-setosa": 
        y1.extend([1,0,0])
    if i == "Iris-versicolor":
         y1.extend([0,1,0])
    if i == "Iris-virginica":
        y1.extend([0,0,1])
        
y = (np.array(y1)).reshape((len(y),3))
############################################


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return ((x * (1-x)))

weights0 = 2 * np.random.random((4,6)) - 1 
weights1 = 2 * np.random.random((6,3)) - 1

# =============================================================================
# print(weights0)
# print(weights1)
# =============================================================================

for i in range(100000):
    #calculating second and third layer output
    outputs1 = sigmoid(np.dot(x, weights0))
    outputs2 = sigmoid(np.dot(outputs1, weights1))
      
    #error calculation and backpropagation
    error2 = y - outputs2
    adjustments2 = error2 * sigmoid_derivative(outputs2)
    error1 = np.dot(adjustments2, weights1.T)
    adjustments1 = error1 * sigmoid_derivative(outputs1)
    
    #update our weights
    weights1 += np.dot(outputs1.T, adjustments2 )
    weights0 += np.dot(x.T, adjustments1)
    
#print(accuracy_score(y, outputs2) * 100)
print(outputs2.shape)
print(outputs2)

#print(outputs)

