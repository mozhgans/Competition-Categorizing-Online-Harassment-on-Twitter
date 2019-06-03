#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 01:23:47 2019
@author: samuel
"""
import numpy as np
import pandas as pd
import preprocessing as pp
import models as md


def main():
    #Step 1: read the data
    trainArch = '/home/samuel/Documents/SIMAH/Data/Train_data_compeition.csv'
    testeArch = '/home/samuel/Documents/SIMAH/Data/Validation_data_competition.csv'
    #trainArchSize = 6374
    control = -1
    print('---------------------------------------------------')
    print('- Welcome to the SIMAH Competition 2019!')
    print('---------------------------------------------------\n ') 
    while control != 0:
        print('Type one of the options below:\n 1 - Read The data set. \n 2 - Pre-process the data sets and run the models. \n 3 - Print the data set.\n press any other key to quit the system.')
        control = int(input('Which option would you like to perform?:'))
        if control == 1:
            trainingData = pp.readData(trainArch)
            testData = pp.readData(testeArch)
            #tweets = tweets.append(testData)
            print('The training and test sets were read!')
        elif control == 2:
            for numFeatures in range(20,55,5):
                trainTweets, featuresList = pp.preProcessTraining(pp.toLowercase(trainingData),numFeatures)
                testTweets = pp.preProcessTest(pp.toLowercase(testData),featuresList)
                md.initModels(trainTweets,testTweets,numFeatures)
        elif control == 3:
            do
        else:
            control = 0
    
if __name__=='__main__':
    main()
    
    
