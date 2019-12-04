install.packages('arules');
install.packages("arulesViz")
library(arulesViz);
data(Groceries);

summary(Groceries)

#Strong Support
apriori(Groceries, parameter= list(support = 0.01, confidence = 0.26)) -> rule1
inspect(head(rule1,5))
inspect(head(sort(rule1, by="support"), 5))


#sTRONG SUPPORT WITH STROG CONFIDENCE WILL MAKE EVEN BETTER PREDICTING

#Strong Confidence
apriori(Groceries, parameter= list(support = 0.001, confidence = 0.55)) -> rule2
inspect(head(rule2,5))
inspect(head(sort(rule2, by="confidence"), 5))


# Strong lift
apriori(Groceries, parameter= list(support = 0.002, confidence = 0.5)) -> rule4
inspect(head(rule4,5))
inspect(head(sort(rule4, by="lift"), 5))


#Minimum length is 5
apriori(Groceries, parameter = list(support = 0.002, confidence = 0.5, minlen = 5)) -> rule3 
inspect(head(sort(rule3, by="confidence"),5))
plot(rule3, method = "grouped")



