#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 01:26:04 2019

"""
import numpy as np
import pandas as pd 
import sklearn as sk  
from sklearn.linear_model import LogisticRegression  
from sklearn import svm  
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.semi_supervised import LabelPropagation

#This function runs the supervised models of Logistic Regression, SVM, Random Forest, and MLP
def initModels(trainData,testData,numFeatures):
    train_x = trainData.iloc[:,:numFeatures]
    train_y = trainData.iloc[:,numFeatures]
    test_x = testData.iloc[:,:numFeatures]
    test_y = testData.iloc[:,numFeatures]#.values.reshape(-1, 1)
    
    print('Results for {} features:'.format(numFeatures))
    #Logistic Regression
    LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(train_x,train_y)  
    preds = LR.predict(test_x)  
    print(preds)
    print('Accuracy of Logistic Regression: {}.'.format(round(LR.score(test_x,test_y), 4))) 


    #Naive Base
    gnb = GaussianNB()
    gnb.fit(train_x,train_y).predict(test_x)
    print('Accuracy of Naive Bayes: {}.'.format(round(gnb.score(test_x, test_y), 4))) 
    #Decision tree:

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_x, train_y)
    print('Accuracy of Decision tree: {}.'.format(round(clf.score(test_x, test_y), 4))) 


    # Random forest:

    RF=RandomForestClassifier(random_state=10)
    RF.fit(train_x, train_y)
    print('Accuracy of Random Forest: {}.'.format(round(RF.score(test_x, test_y), 4)))
    
    #Linear SVM
    SVM = svm.LinearSVC()  
    SVM.fit(train_x, train_y).predict(test_x)  
    print('Accuracy of Linear SVM: {}.'.format(round(SVM.score(test_x,test_y), 4)))  
    
    #Gaussian SVM
    GSVM = svm.SVC()
    GSVM.fit(train_x, train_y).predict(test_x)  
    print('Accuracy of Gaussian SVM: {}.'.format(round(GSVM.score(test_x,test_y), 4))) 
    
    #Poly SVM
    POLYSVM = svm.SVC(kernel='poly',degree=2)
    POLYSVM.fit(train_x, train_y).predict(test_x)  
    print('Accuracy of Polynomial SVM: {}.'.format(round(POLYSVM.score(test_x,test_y), 4)))
    
    #MLP
    NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)  
    NN.fit(train_x, train_y).predict(test_x) 
    print('Accuracy of MLP: {}.'.format(round(NN.score(test_x,test_y), 4)))
    
    #AdaBoost
    adb = AdaBoostClassifier(n_estimators=100, random_state=0)
    adb.fit(train_x, train_y).predict(test_x)  
    print('Accuracy of Adaboost: {}.'.format(round(adb.score(test_x,test_y), 4)))
    
    
    
