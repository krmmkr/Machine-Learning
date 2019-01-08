# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 14:08:01 2018

@author: kohul
"""
import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

filename = 'C:/Users/kohul/OneDrive/Documents/ML/student-mat.csv'
data = pandas.read_csv(filename, sep =';')

def SingleCheck(data):
    for col in data.columns:
        if len(data[col].unique()) == 1:
            data = data.drop(col, axis=1)
            print(col)
    return data

def plot(iters, cost):
    #plot the cost
    fig, ax = plt.subplots()  
    ax.plot(numpy.arange(iters), cost)  
    ax.set_xlabel('Iterations')  
    ax.set_ylabel('Cost')

#####Creating Dummies###########
def createDummy(data):
    data = pandas.get_dummies(data, drop_first=True)
#    for col in data.columns:
#        if isinstance(data[col][1], str):
#            dummy = pandas.get_dummies(data[col])
#            for dcol in dummy.columns:
#                dummy = dummy.rename(columns = {dcol:col+":"+dcol})
#            data = pandas.concat([data.drop(col, axis=1), dummy.drop(dummy.columns[-1],axis=1)], axis=1)
    return data

#computecost
def computeCost(X,y,beta, regtype):
    #tobesummed = numpy.power(((X @ beta.T)-y),2)
    #return numpy.sum(tobesummed)/(2 * len(X))
    if regtype == 'logistic':
        ycap = 1/(1+numpy.power(2.71828,-(X @ beta.T)))
        return numpy.sum(y*numpy.log(ycap)+(1-y)*numpy.log(1-ycap))/len(X)*-1         
#        return numpy.sum(numpy.power((ycap-y),2))/(2 * len(X))
    if regtype == 'linear':
        return numpy.sum(numpy.power(((X @ beta.T)-y),2))/(2 * len(X))

#gradient descent
def gradientDescent(X,y,beta,iters,alpha, regtype):
    cost = numpy.zeros(iters)
    for i in range(iters):
        beta = beta - (alpha/len(X)) * numpy.sum(X * (X @ beta.T - y), axis=0)
        cost[i] = computeCost(X, y, beta, regtype)
    return beta,cost

#compute Ycap
def computeYcap(X,beta, regtype):
    if regtype == 'logistic':
        return 1/(1+numpy.power(2.71828,-(X @ beta.T)))
    if regtype == 'linear':
        return X @ beta.T
#    ycap = X @ beta.T
#    return ycap

#def computeProbability(X,beta):
#    y = X @ beta.T
#    ycap = 1/(1+numpy.power(2.71828,-y))
#    return ycap

def standardize(var):
    var = (var - var.mean())/var.std()
    return var

def confusionMatrix(yprob, yclass, cutoff, t):
    ycap = (numpy.where(yprob>cutoff,1,0))
    TP = numpy.sum(numpy.logical_and(ycap == 1, yclass == 1))
    TN = numpy.sum(numpy.logical_and(ycap == 0, yclass == 0))
    FP = numpy.sum(numpy.logical_and(ycap == 1, yclass == 0))
    FN = numpy.sum(numpy.logical_and(ycap == 0, yclass == 1))
    print(t)
    print("TP:{} TN:{} FP:{} FN:{}".format(TP,TN,FP,FN))
    print("Accuracy:{}".format((TP+TN)/(TP+TN+FP+FN)))
    print("Sensitivity:{}".format((TP)/(TP+FN)))
    print("Specificity:{}".format((TN)/(TN+FP)))

data_new = data.drop(['G1','G2','G3'],axis=1)
data_new = SingleCheck(data_new)
###################################Linear Regression All features###################################
print("\nLinear Regression ALL Features\n")
X = createDummy(data_new)
ones = numpy.ones([X.shape[0],1])
print(X.columns)
X = numpy.concatenate((ones,X),axis=1)

#set hyper parameters
alpha = 0.001
iters = 1000
beta = numpy.zeros([1,40])

y=data.iloc[:,32:33].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#running the gd and cost function
b,cost = gradientDescent(X_train,y_train,beta,iters,alpha, 'linear')
print(b)

finalCost = computeCost(X_train,y_train,b,'linear')
print(finalCost)

ycap_train = computeYcap(X_train,b, 'linear')
ycap_test = computeYcap(X_test,b, 'linear')
finalCost = computeCost(X_test,y_test,b, 'linear')
print(finalCost)
plot(iters, cost)
###############################Logistic Regression All features##########################################
print("\nLogistic Regression ALL Features\n")
ylogi = (numpy.where(y>numpy.median(y),1,0))
X2_train, X2_test, y2_train, y2_test = train_test_split(X, ylogi, test_size=0.3, random_state=1)

#set hyper parameters
alpha = 0.003
iters = 10000
beta = numpy.zeros([1,40])
#running the gd and cost function
b,cost = gradientDescent(X2_train,y2_train,beta,iters,alpha, 'logistic')
print(b)

finalCost = computeCost(X2_train,y2_train,b,'logistic')
print(finalCost)

ycap2_train = computeYcap(X2_train,b,'logistic')
ycap2_test = computeYcap(X2_test,b,'logistic')
finalCost = computeCost(X2_test,y2_test,b,'logistic')
print(finalCost)

plot(iters, cost)

confusionMatrix(ycap2_train, y2_train, 0.6, 'Training Data')
confusionMatrix(ycap2_test, y2_test, 0.6, 'Test Data')

#######################Linear Regression 10 Random features###########################################################
print("\nLinear Regression 10 Random Features\n")
#set hyper parameters
alpha = 0.01
iters = 10000
beta = numpy.zeros([1,11])
X = createDummy(data_new).sample(10,axis=1,random_state=1)
print(X.columns)
X = numpy.concatenate((ones,X),axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
beta = numpy.zeros([1,11])
#running the gd and cost function
b,cost = gradientDescent(X_train,y_train,beta,iters,alpha, 'linear')
print(b)

finalCost = computeCost(X_train,y_train,b,'linear')
print(finalCost)

ycap_train = computeYcap(X_train,b, 'linear')
ycap_test = computeYcap(X_test,b, 'linear')
finalCost = computeCost(X_test,y_test,b, 'linear')
print(finalCost)

plot(iters, cost)

#######################################Logistic Regression for 10 Random feature###########################
print("\nLogistic Regression 10 Random Features\n")
ylogi = (numpy.where(y>numpy.median(y),1,0))
X2_train, X2_test, y2_train, y2_test = train_test_split(X, ylogi, test_size=0.3, random_state=1)

#set hyper parameters
alpha = 0.1
iters = 10000
beta = numpy.zeros([1,11])
#running the gd and cost function
b,cost = gradientDescent(X2_train,y2_train,beta,iters,alpha, 'logistic')
print(b)

finalCost = computeCost(X2_train,y2_train,b,'logistic')
print(finalCost)

ycap2_train = computeYcap(X2_train,b,'logistic')
ycap2_test = computeYcap(X2_test,b,'logistic')
finalCost = computeCost(X2_test,y2_test,b,'logistic')
print(finalCost)

plot(iters, cost)

confusionMatrix(ycap2_train, y2_train, 0.6, 'Training Data')
confusionMatrix(ycap2_test, y2_test, 0.6, 'Test Data')

#####################################Linear Regression 10 Selected features#####################################################
#X = data_new.drop(['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','failures'],axis=1)
#X = data_new.loc[:,['age','Medu','Fedu','traveltime','studytime','famrel','goout','freetime','health','absences']].values
#set hyper parameters
print("\nLinear Regression 10 Selected Features\n")
alpha = 0.01
iters = 10000
beta = numpy.zeros([1,11])
X = createDummy(data_new)
X=X.loc[:,['Medu', 'studytime', 'Fjob_teacher', 'Fedu', 'goout',
       'freetime', 'famrel', 'schoolsup_yes', 'health', 'absences']]
X = numpy.concatenate((ones,X),axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
beta = numpy.zeros([1,11])
#running the gd and cost function
b,cost = gradientDescent(X_train,y_train,beta,iters,alpha, 'linear')
print(b)

finalCost = computeCost(X_train,y_train,b,'linear')
print(finalCost)

ycap_train = computeYcap(X_train,b, 'linear')
ycap_test = computeYcap(X_test,b, 'linear')
finalCost = computeCost(X_test,y_test,b, 'linear')
print(finalCost)

plot(iters, cost)
#######################################Logistic Regression for 10 Selected feature###########################
print("\nLogistic Regression 10 Selected Features\n")
ylogi = (numpy.where(y>numpy.median(y),1,0))
X2_train, X2_test, y2_train, y2_test = train_test_split(X, ylogi, test_size=0.3, random_state=1)

#set hyper parameters
alpha = 0.009
iters = 1000
beta = numpy.zeros([1,11])
#running the gd and cost function
b,cost = gradientDescent(X2_train,y2_train,beta,iters,alpha, 'logistic')
print(b)

finalCost = computeCost(X2_train,y2_train,b,'logistic')
print(finalCost)

ycap2_train = computeYcap(X2_train,b,'logistic')
ycap2_test = computeYcap(X2_test,b,'logistic')
finalCost = computeCost(X2_test,y2_test,b,'logistic')
print(finalCost)

plot(iters, cost)

confusionMatrix(ycap2_train, y2_train, 0.5, 'Training Data')
confusionMatrix(ycap2_test, y2_test, 0.59, 'Test Data')