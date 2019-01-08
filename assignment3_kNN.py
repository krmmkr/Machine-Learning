#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 

#####Creating Dummies###########
def createDummy(data):
    data = pd.get_dummies(data, drop_first=True)
    return data
###############################
def Check(data):
    for col in data.columns:
        if len(data[col].unique()) == len(data[col]):
            data = data.drop(col, axis=1)
            print(col)
    return data

##########################################
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred))
###############################################

dataset = 'C:/Users/kohul/OneDrive/Documents/ML/student-mat.csv'
data = pd.read_csv(dataset, sep=';')

#Data Processing
Y=data.iloc[:,32:33].values
Y = (np.where(Y>10,1,0))
data = data.drop(['G1','G2','G3'],axis=1)
data = Check(data)
X = createDummy(data)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3, random_state=42)

accuracy = []
for i in range(1,10):
	#Create KNN Classifier
	knn = KNeighborsClassifier(n_neighbors=i, p=2)

	#Train the model using the training sets
	knn.fit(Xtrain, ytrain)

	#Predict the response for test dataset
	y_pred = knn.predict(Xtest)
	accuracy.append(metrics.accuracy_score(ytest, y_pred))

knn = KNeighborsClassifier(n_neighbors=5, p=1)
knn.fit(Xtrain, ytrain)
y_pred = knn.predict(Xtest)
cal_accuracy(y_pred, ytest)

plt.plot(accuracy)
plt.show()

# step 1: get the data and define all the usual variables
dataset2 = 'C:/Users/kohul/OneDrive/Documents/ML/titanic.csv'
data2 = pd.read_csv(dataset2, sep=',')

#Data Processing
Y2=data2.iloc[:,1:2].values
data2 = data2.drop(['Survived','Name','Ticket','Cabin'],axis=1)
data2 = Check(data2)
X2 = createDummy(data2)
Xtrain2, Xtest2, ytrain2, ytest2 = train_test_split(X2, Y2, test_size=0.3, random_state=10)

accuracy = []
for i in range(1,10):
	#Create KNN Classifier
	knn = KNeighborsClassifier(n_neighbors=i, p=1)

	#Train the model using the training sets
	knn.fit(Xtrain2, ytrain2)

	#Predict the response for test dataset
	y_pred2 = knn.predict(Xtest2)
	accuracy.append(metrics.accuracy_score(ytest2, y_pred2))

knn = KNeighborsClassifier(n_neighbors=4, p=1)
knn.fit(Xtrain2, ytrain2)
y_pred2 = knn.predict(Xtest2)
cal_accuracy(y_pred2, ytest2)

plt.plot(accuracy)
plt.show()
#print("Accuracy:",metrics.accuracy_score(ytest, y_pred))