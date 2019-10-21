import pandas
import random
import numpy as np
import seaborn as sns
from scipy.stats import entropy


names=['sepal_length', 'sepal_breadth', 'petal_length', 'petal_breadth', 'class' ]
data = pandas.read_csv(r"C:\Users\Rochak\Desktop\Machine Learning Exercise\iris.data", names=names, na_values=["?"], skipinitialspace = True, header=None)

# =============================================================================
# print(data.columns);
# print(data.head);
# print(data.info)
# print(data.isna().any())
# #print(data[data.isna().any(axis=1)] )
# 
# =============================================================================


def train_test_split(data , test_size):
    indexes = data.index.tolist()   
    if test_size < 1: test_size = int(test_size * 100)
    data = data.sample(frac = 1).reset_index(drop=True)
    random_test_indexes = random.sample(population = indexes, k = test_size) 
    test = data.iloc[random_test_indexes]  
    train = data.drop(random_test_indexes)
    xTrain = train.iloc[0:, :-1]
    yTrain = train.iloc[0:, -1]
    xTest = test.iloc[0:, :-1]
    yTest = test.iloc[0:, -1]
    return xTrain, xTest, yTrain, yTest


def decision_tree_algorithm():
    pass

def check_purity(column_name):
# =============================================================================
# # This is if we have used zip, tuple will be converted to list 
#    column_name = np.array(column_name)
#     column_name = column_name[0:, -1]
# =============================================================================
    #print(np.unique(column_name))
    unique_classes = np.unique(column_name)
    if len(unique_classes) == 1:
        return True
    else:
        return False
    
def classify_data(column_name):
    unique_classes, unique_classes_count = np.unique(column_name, return_counts = True)
    index = unique_classes_count.argmax()
    return unique_classes[index]


def get_entropy(yTrain, base=None):
  value,counts = np.unique(yTrain, return_counts=True)
  print(value, counts)
  return entropy(counts, base=base)

def get_potential_split(x_training_data):
    potential_splits = {}
    column_n = (x_training_data.shape[1])
    for column_index in range(column_n):
        potential_splits[column_index] = []
        values = x_training_data[:, column_index]
        unique_values = np.unique(values)
        
        for index in range(len(unique_values)):
            if index != 0:
                current_value = unique_values[index]
                previous_value = unique_values[index-1]
                potential_split = (current_value + previous_value) /2      
                potential_splits[column_index].append(potential_split)
    
    return potential_splits


def split_data():
    pass    





xTrain, xTest, yTrain, yTest = train_test_split(data, test_size = 0.3)
# =============================================================================
# print(check_purity(list(zip(xTrain,yTrain))))
# =============================================================================
purity = check_purity(yTest)
classify = (classify_data(yTrain))
potential_splits = get_potential_split(xTrain.values)
#print(potential_splits)
sns.lmplot(data = xTrain, x = "petal_breadth", y ="petal_length")
print(get_entropy(yTrain))




