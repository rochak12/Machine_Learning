library(e1071)

# columns = [" "]
data <-read.csv(file = "C:/Users/Rochak/Desktop/Machine Learning Exercise/vowel.csv", head=TRUE)

print(data)
# dim(data)
allRows  <- 1:nrow(data)
testRows <- sample(allRows, trunc(length(allRows) * 0.3))

# The test set contains all the test rows
VowelTest <- data[testRows,]

# The training set contains all the other rows
VowelTrain <- data[-testRows,]

# Train an SVM model
# Tell it the attribute to predict vs the attributes to use in the prediction,
# the training data to use, and the kernal to use, along with its hyperparameters.
# Please note that "Class~." contains a tilde character, rather than a minus
lengths(lapply(data, unique))
model <- svm(Class~., data = VowelTrain, kernel = "radial", gamma = 0.1, cost = 40, type ="C")

# Use the model to make a prediction on the test set
# Notice, we are not including the last column here (our target)
prediction <- predict(model, VowelTest[,-13])

# Produce a confusion matrix
confusionMatrix <- table(pred = prediction, true = VowelTest$Class)

# Calculate the accuracy, by checking the cases that the targets agreed
agreement <- prediction == VowelTest$Class
accuracy <- prop.table(table(agreement))

# Print our results to the screen
print(confusionMatrix)
print(accuracy)




