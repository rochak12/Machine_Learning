import pandas


from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report



names = ["Party", "handicaped_infants" , "water_project_cost_sharing", "budget_resolution", "physician_fee_freeze", "el-salvador_aid", "relid_in_school",
         "settilite_ban", "aid_niccaruaga", "mx-missile", "immigration", "synfuels", "educatiopn", 
         "superfaund" , "crime" , "duty_free", "south_Africa"];
data = pandas.read_csv(r"C:\Users\Rochak\Desktop\Machine Learning Exercise\house-votes.data", names = names, delimiter = "," ,na_values=["?"], skipinitialspace = True)


x = data.iloc[0:, 1:]
y = data.iloc[0:, 0]
#print(data.columns)



print(data.isna().any())
#print(data[data.isna().any(axis=1)] )



#data.dropna()
# =============================================================================
# for datai in data:
#     data[datai] = data[datai].fillna("unknown")
# print(data)
# =============================================================================

#data.fillna(data.mean(), inplace=True)  #This goes after encoding




print(data.isnull().sum())
lae = data.isna().any()
#print("list is\n" ,  lae)
print(data)



columns_with_object = x.select_dtypes(include=["object"]).columns
#print("here" ,columns_with_object)

# =============================================================================
# for column in columns_with_object:    
#     print(data[column].value_counts())
# =============================================================================

#Encoding
for column in columns_with_object:
    data[column] = data[column].astype('category')
    temp = column+"_cat"
    data[temp] = data[column].cat.codes
  
    
columns_with_object = data.select_dtypes(include=["object"]).columns
#print(columns_with_object)    
#print(data)

   
data = data.drop(columns=[ "handicaped_infants" , "water_project_cost_sharing", "budget_resolution", "physician_fee_freeze", "el-salvador_aid", "relid_in_school",
         "settilite_ban", "aid_niccaruaga", "mx-missile", "immigration", "synfuels", "educatiopn", 
         "superfaund" , "crime" , "duty_free", "south_Africa"])

print(data.isnull().sum())
lae = data.isna().any()    
x = data.iloc[0:, 1:]   



xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = 0.3, random_state = 0)

classifier = DecisionTreeClassifier(criterion = "entropy", random_state= 100, max_depth = 4, min_samples_leaf = 1)
classifier.fit(xTrain, yTrain)
tree.plot_tree(classifier.fit(xTrain, yTrain)) 
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
Class = (np.unique(data.Party))
tree.export_graphviz(classifier, out_file=dot_data, 
                     feature_names=data.columns[1:] ,
                     class_names= Class) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 

graph[0].write_pdf("house_votes.pdf") 
#Image(graph[0].create_png()) 
