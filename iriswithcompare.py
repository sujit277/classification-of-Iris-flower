from sklearn.datasets import load_iris
iris=load_iris()
#print(iris)
##input  and output
X=iris.data ## input
Y=iris.target###output
#print(X.shape)#(150,4)
#print(Y.shape)#(150,)
##split the data for training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
print(X_train.shape)#(120,4)
print(X_test.shape)#(30,4)
print(Y_train.shape)#(120,)
print(Y_test.shape)#(30,)
##########################################
#Implement KNN MODEL
###################################
######### create  a model  KNN
## KNN --Knearest neighbor algorithm
from sklearn.neighbors import KNeighborsClassifier
K=KNeighborsClassifier(n_neighbors=5)
###train the model
K.fit(X_train,Y_train)
####test  the model
Y_pred_knn=K.predict(X_test)

###find accuracy
from sklearn.metrics import accuracy_score

acc_knn=accuracy_score(Y_test,Y_pred_knn)
acc_knn=round(acc_knn*100,2)
print("accuracy in knn is",acc_knn)

####predict for  a new  flower
print(K.predict([[4,2,3,1]]))#1

#0-----setosa
#1----versiclor
#2---virginica
##############################################
#Implement Logistic Regression
##########################################3
from sklearn.linear_model import LogisticRegression
L=LogisticRegression()
##train the model
L.fit(X_train,Y_train)
##test the model
Y_pred_lg=L.predict(X_test)
#find accuracy
from sklearn.metrics import accuracy_score
acc_lg=accuracy_score(Y_test,Y_pred_lg)
acc_lg=round(acc_lg*100,2)
print("accuracy in logistic regression is",acc_lg)

####predict for  a new  flower
print(L.predict([[4,2,3,1]]))#

#################################
#implement Decision tree classifier
############################################
from sklearn.tree import DecisionTreeClassifier
D=DecisionTreeClassifier()
##train the model
D.fit(X_train,Y_train)
##test the model
Y_pred_dt=D.predict(X_test)
#find accuracy
from sklearn.metrics import accuracy_score
acc_dt=accuracy_score(Y_test,Y_pred_dt)
acc_dt=round(acc_dt*100,2)
print("accuracy in decision tree is",acc_dt)

####predict for  a new  flower
print(D.predict([[4,5,6,1]]))#

#############################################
#Implement Naive Bayes Algorithm
######################################
## Implement Naive Bayes  Algorithm
from sklearn.naive_bayes  import GaussianNB
N=GaussianNB()
##train the model
N.fit(X_train,Y_train)
##test the model
Y_pred_nb=L.predict(X_test)
#find accuracy
from sklearn.metrics import accuracy_score
acc_nb=accuracy_score(Y_test,Y_pred_nb)
acc_nb=round(acc_nb*100,2)
print("accuracy in naive bayes is",acc_nb)

####predict for  a new  flower
print(N.predict([[2,2,2,0.2]]))#
#################################################
###########COMPARE   THE  MODELS
########################################
import matplotlib.pyplot as plt
models=['KNN','LG','DT','NB']
accuracy=[acc_knn,acc_lg,acc_dt,acc_nb]
plt.bar(models,accuracy,color=['green','yellow','blue','orange'])
plt.xlabel("MODElS")
plt.ylabel("ACCURACY")
plt.show()
