# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import scipy
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import KNeighborsClassifier
#Loading the datasets

m = scipy.io.loadmat("D:\Learning\Umass\ML 689\Individual mini projects\SSL,set=9,data.mat")
X=m['X']
y=m['y']
n = scipy.io.loadmat("D:\Learning\Umass\ML 689\Individual mini projects\SSL,set=9,splits,labeled=10.mat")

# Taking the labeled and unlabeled data into an array
yL=np.array(n['idxLabs'])
yU=np.array(n['idxUnls'])

# Creating the Affinity/Adjacency matrix
adjacency_matrix = kneighbors_graph(X.toarray(),5, mode='connectivity', include_self=True).toarray()
rowsum = adjacency_matrix.sum(axis=1)

# Degree matrix formed by setting diagonal elements to the row sums of Affinity matrix
D=np.diag(rowsum)
Dinv= np.linalg.inv(D)

# Generating the Random Walk matrix
R=Dinv.dot(adjacency_matrix)
#Initialize a matrix of size 1500x1 to fill the indices corresponding to each row of labeled matrix 
#with the labels in output y
Y0=np.zeros((1500,1))
  
# User defined fucntion to fill the Y0 matrix       
def fill_Y0(iteration_number,label_indices,empty_Y0,labeled_matrix):
    for i in range(0,label_indices[iteration_number].shape[0]):
        index=label_indices[iteration_number][i]
        empty_Y0[index]=labeled_matrix[index]
    return empty_Y0

#Iterating the dot product of randomwalk matrix with the filled Y0 matrix to propagate labels
def propagate_labels(random_walk,filled_Y0):
    propagated_labelmatrix=random_walk.dot(filled_Y0)
    for k in range(0,10):
        propagated_labelmatrix=random_walk.dot(propagated_labelmatrix)
    return propagated_labelmatrix

 # User defined function to check the accuracy of each label propagation iteration
 # by comparing the propagated labels with the actual ones in output y
def check_accuracy(labeled_matrix,propagated_labelmatrix):
    count=0
    accuracy=0
    for l in range(0,propagated_labelmatrix.shape[0]):
        if propagated_labelmatrix[l]>0:
            propagated_labelmatrix[l]=1
        elif propagated_labelmatrix[l]<0:
            propagated_labelmatrix[l]=-1
    count=(propagated_labelmatrix==labeled_matrix).sum()
    accuracy=count/float((labeled_matrix).shape[0])
    return accuracy
# User defined function to test the SSL algorithm for label propagation
def iterative_testing(label_indices,random_walk,empty_Y0,labeled_matrix):
    accuracy_list=[]
    for j in range(0,label_indices.shape[0]):
        convergence_accuracy=0
        filled_Y0=fill_Y0(j,label_indices,empty_Y0,labeled_matrix)
        propagated_labelmatrix=propagate_labels(random_walk,filled_Y0)
        convergence_accuracy= check_accuracy(labeled_matrix,propagated_labelmatrix)
        accuracy_list.append(convergence_accuracy)
        print "Accuracy for iteration #"+str(j)+" is : "+ str(convergence_accuracy)
    average_accuracy=sum(accuracy_list)/float(label_indices.shape[0])
    print "Average Accuracy for this algorithm is "+ str(average_accuracy)
    return average_accuracy
    
    
result1=iterative_testing(yL,R,Y0,y)

# Comparing with K nearest Neighbours Algorithm
 
def knn_labelprop(x,y,X,labeled_matrix):
    neigh = KNeighborsClassifier(n_neighbors=5)  
    neigh.fit(x,y)
    propagated_labelmatrix= neigh.predict(X)
    knn_accuracy= check_accuracy(labeled_matrix,propagated_labelmatrix)
    print "Accuracy for KNN algorithm is "+ str(knn_accuracy)
    return knn_accuracy
c=y.reshape(1500)
result2=knn_labelprop(X.toarray()[0:10],c[0:10],X.toarray(),c)

percentage = 100*abs(result1-result2)/float(result1)
print "Semi supervised Label propagation gives " +str(float("{0:.2f}".format(percentage)))+ "% more accuracy than KNN"

