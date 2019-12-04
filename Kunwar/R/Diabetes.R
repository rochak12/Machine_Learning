library(e1071)


data <-read.csv(file = "C:/Users/Rochak/Desktop/Machine Learning Exercise/pima-indians-diabetes-database/diabetes.csv", head=TRUE)
print(data)


allRows  <- 1:nrow(data)
testRows <- sample(allRows, trunc(length(allRows) * 0.3))

# The test set contains all the test rows
diabetesTest <- data[testRows,]

# The training set contains all the other rows
diabetesTrain <- data[-testRows,]

# Train an SVM model
# Tell it the attribute to predict vs the attributes to use in the prediction,
# the training data to use, and the kernal to use, along with its hyperparameters.
# Please note that "Outcome~." contains a tilde character, rather than a minus
model <- svm(Outcome~., data = diabetesTrain, kernel = "radial", gamma = 0.01, cost = 10, type ="C")
lengths(lapply(data, unique))


# Use the model to make a prediction on the test set
# Notice, we are not including the last column here (our target)
prediction <- predict(model, diabetesTest[,-9])

# Produce a confusion matrix
confusionMatrix <- table(pred = prediction, true = diabetesTest$Outcome)

# Calculate the accuracy, by checking the cases that the targets agreed
agreement <- prediction == diabetesTest$Outcome
accuracy <- prop.table(table(agreement))

# Print our results to the screen
print(confusionMatrix)
print(accuracy)





