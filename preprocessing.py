#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 00:11:16 2019
@author: samuel
"""
import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

def readData(file):
    df = pd.read_csv(file,sep=',', header=0, index_col=0)
    return df

def preProcessTraining(dataFrame,numFeatures):
    answer = []
    wordnet_lemmatizer = WordNetLemmatizer()
    vectorizer = TfidfVectorizer(stop_words='english',max_features=numFeatures)
    corpus = []
    for index, row in dataFrame.iterrows():
        sentence_words = nltk.word_tokenize(row['tweet_content'])
        word_list = []
        for word in sentence_words:
            word = wordnet_lemmatizer.lemmatize(word)
            if 'rt' not in word:
                word_list.append(word)
                word_list.append(' ')
            else: pass
        corpus.append(''.join(word_list)) 
    vectors = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names()
    print('Lenght of features list: {}.'.format(len(features)))
    print(features)
    vectors = vectors.toarray()
    for index, row in dataFrame.iterrows():
        vec = np.append(vectors[index],[row['harassment'],row['IndirectH'],row['PhysicalH'],row['SexualH']])
        answer.append(vec)
    newDataFrame = pd.DataFrame(answer)
    return newDataFrame, features

def preProcessTest(df,featuresList):
    answer = []
    i = range(0,len(featuresList))
    featuresNames = dict(zip(featuresList, i))
    wordnet_lemmatizer = WordNetLemmatizer()
    vectorizer = TfidfVectorizer(stop_words='english',max_features=len(featuresList),vocabulary=featuresNames)
    corpus = []
    for index, row in df.iterrows():
        sentence_words = nltk.word_tokenize(row['tweet_content'])
        word_list = []
        for word in sentence_words:
            word = wordnet_lemmatizer.lemmatize(word)
            if 'rt' not in word:
                word_list.append(word)
                word_list.append(' ')
            else: pass
        corpus.append(''.join(word_list)) 
    vectors = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names()
    print('Lenght of features list: {}.'.format(len(features)))
    #print(features)
    vectors = vectors.toarray()
    #print(df)
    for index, row in df.iterrows():
        vec = np.append(vectors[index-6374],[row['harassment'],row['IndirectH'],row['PhysicalH'],row['SexualH']])
        answer.append(vec)
    newDataFrame = pd.DataFrame(answer)
    return newDataFrame

def toLowercase(dataFrame):
    for index, row in dataFrame.iterrows():
        dataFrame.loc[index,['tweet_content']] = row['tweet_content'].lower()
    return dataFrame

def dimReduction(dataFrame,originalDim):
    return
