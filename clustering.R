# library(datasets)
# myData = state.x77
# summary(myData)
# 
# 
# distance = dist(as.matrix(myData))
# hc = hclust(distance)
# plot(hc)
# 
# 
# 
# data_scaled = scale(myData)
# distance = dist(as.matrix(data_scaled))
# hc = hclust(distance)
# plot(hc)
# 
# 
# #shape in r
# dim(data_scaled)
# 
# #deleting area 
# new_myData = data_scaled[ , -8]
# summary(new_myData)
# distance = dist(as.matrix(new_myData))
# hc = hclust(distance)
# plot(hc)
# 
# 
# #using only frost
# new_myData2 = data_scaled[ , 7]
# summary(new_myData2)
# distance = dist(as.matrix(new_myData2))
# hc2 = hclust(distance)
# plot(hc2)



#K-mean starts
library(datasets)
myData = state.x77
data_scaled = scale(myData)
myData = data_scaled

#list for within_clustor and accross_clustor
within_clustor = vector(mode = "list", length = 25)
total_within = vector(mode = "list" , length = 25)

for (i in 1 : 25){
  # Cluster into k=3 clusters:
  myClusters = kmeans(myData, i)
  
  # Summary of the clusters
  summary(myClusters)
  
  # Centers (mean values) of the clusters
  myClusters$centers
  
  # Cluster assignments
  myClusters$cluster
  
  # Within-cluster sum of squares and total sum of squares across clusters
  myClusters$withinss
  total_within[i] <-  myClusters$tot.withinss
}


dim(total_within)
plot(1:25, total_within[i])


# Plotting a visual representation of k-means clusters
library(cluster)
clusplot(myData, myClusters$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)




table = NULL;
for (i in 1:10) {
  table[i] = i * 2
}


