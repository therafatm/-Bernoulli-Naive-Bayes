#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import logging
import sys
from time import time

class MyBayesClassifier():
    def __init__(self, smooth=1):
        self._smooth = smooth
        self._feat_prob = {}
        self._class_prob = {}
        self._classDict = {}
        self._featureMap = {}
        self._Ncls = 0
        self._Nfeat = 0

    def computeClassProbability(self, totalRecords):
        for x in self._classDict.keys():
            self._class_prob[x] = (self._classDict[x] + self._smooth) / float(totalRecords + (2**self._smooth)) # 2 because bernoullli (2 options)

    def train(self, X, y):

        classDict = self.makeClasses(y)
        numFeatures = X.shape[1]
        classMap = {k: {"count": 0, "probability": 0} for k in classDict.keys()}
        featureMap = {k: classMap for k in range(0,numFeatures)}
        #featureMap contains a dictionary for every feature
        #which contains a dictionary for every class, and stores zero_count given class

        #go through the array and add all the counts for 0 given Class C
        for row in range(0, X.shape[0]):
            for column in range(0, X.shape[1]):
                val = 0 | X[row][column]
                if(val == 0):
                    correspondingClassValue = y[row]
                    featureMap[column][correspondingClassValue]["count"] += 1
                if(row == X.shape[0] - 1):
                    # calculate probablity and account for alpha
                    featureMap[column][correspondingClassValue]["probability"] = (featureMap[column][correspondingClassValue]["count"] + self._smooth) / float(classDict[correspondingClassValue] + (2**self._smooth))

        self._featureMap = featureMap
        self._Ncls = len(classDict.keys())
        self._Nfeat = numFeatures
        self._classDict = classDict
        self._class_prob = self.computeClassProbability(X.shape[0])

    def getPointProbability(self, featureName, columnName, className, value):

        retrieved = self._featureMap[fname][str(className)]["probability"]
        if(val == 1):
            retrieved = 1 - retrieved
        return retrieved

    def predict(self, X):

        results = np.zeros([X.shape[0],1])
        for row in range(0, X.shape[0]):
            maxVal = 0
            finalChoice = -1
            for className in self._classDict.keys():
                partialTotal = self._class_prob[str(className)]
                for column in range(0, X.shape[1]):
                    partialTotal *= self.getPointProbablity(row,column,className,X[row][column])
                #set maxVal/result
                if(maxVal<partialTotal):
                    maxVal = partialTotal
                    finalChoice = className
            results[row] = finalChoice

        return results

    def makeClasses(self, y):
        classes = {}
        for i in range(0,len(y)):
            val = y[i]
            if(classes.has_key(val) == False):
                classes[val] = 1
            else:
                classes[val] += 1
        return classes

categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
remove = ('headers', 'footers', 'quotes')

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)
print('data loaded')

y_train, y_test = data_train.target, data_test.target

print("Extracting features from the training data using a count vectorizer")
t0 = time()

vectorizer = CountVectorizer(stop_words='english', binary=True)#, analyzer='char', ngram_range=(1,3))
X_train = vectorizer.fit_transform(data_train.data).toarray()
X_test = vectorizer.transform(data_test.data).toarray()
feature_names = vectorizer.get_feature_names()

alpha = 1
clf = MyBayesClassifier(alpha)
clf.train(X_train,y_train)
y_pred = clf.predict(X_test)
print 'alpha=%i accuracy = %f' %(alpha, np.mean((y_test-y_pred)==0))