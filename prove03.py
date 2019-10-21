import sklearn
import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression





names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "classvalue"]
data = pandas.read_csv(r"C:\Users\Rochak\Desktop\Machine Learning Exercise\car.data", names=names, na_values=["?"], skipinitialspace = True, header=None)
# =============================================================================
#print(data)
#print(data.dtypes)
#print("Median age" , data.doors.median())
#print(data.doors.value_counts())
#print(data.columns)
#[print(column) for column in data.columns]
#print(data.head())
# =============================================================================




#print(data.isna().any())
#print(data[data.isna().any(axis=1)] )

#data.workclass = data.workclass.fillna("unknown")
#data.native_country = data.native_country.fillna("unknown")
#data.occupation = data.occupation.fillna("unknown")

#lae = data.isna().any()
#print("list is\n" ,  lae)


columns_with_object = data.select_dtypes(include=["object"]).columns
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
print(columns_with_object)    
#print(data)

   
data = data.drop(columns=["buying", "maint", "doors", "persons", "lug_boot", "safety", "classvalue"])
#print(data)
      


    
x = data.iloc[0:, :-1]
y = data.iloc[0:, -1]
x,y = shuffle(x,y, random_state = 0)
xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = 0.3, random_state = 0)
classifier = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski' , p=2)
classifier.fit(xTrain, yTrain)
predictions = classifier.predict(xTest)
print(accuracy_score(yTest, predictions) * 100)

