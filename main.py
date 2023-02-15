# Project 1: Density Estimation and Classification - SML 2022


"""
Created on Sun May 29 2022
@author: Aishwarya Balaji Rao
"""


import pandas as pd
import numpy as np
import scipy.io
from scipy.stats import norm
import math
from statistics import mean


class NaiveBayesClassifier(object):

    """ 
    Describing objects to initialize in the NaiveBayes class
    Inputs taken: Train dataset labels - TrY
    Variables defined: mean, preidctions, standard deviation and train labels
    """

    def __init__(self, N_labels):
        self.mean = []
        self.totalPredictions = []
        self.std = []
        self.N_labels = N_labels
        
    def fit_classifier(self, data, labels):

        """ 
        Computing features for the datasets: mean and standard deviation and training
        the naive bayes model
        Inputs taken: Train X (training images) & Train Y (labels) datasets 
        """

        self.set_prior(labels)
        for i in range(self.N_labels):
            self.mean.append(data[labels == i].mean(axis = 0))
            self.std.append(data[labels == i].std(axis = 0))

        # Making single arrays by stacking vertically
        self.mean = np.vstack(self.mean)
        self.std = np.vstack(self.std)
        print ("Mean is: ", mean(self.mean.flatten()))
        print("STD is: ", mean(self.std.flatten()))

        # Assume lowest value for std = o to avoid undefined values.
        self.std[self.std == 0] = 0.0000001

    def set_prior(self, labels):

        """
        Set each label's prior probability by counting how often it occurs in 
        training data divided by the total number of training samples.
        Inputs taken: The training set labels.
        """

        priorCounts = np.unique(labels, return_counts = True)[1]
        self.prior = priorCounts/np.sum(priorCounts)
        print("Prior", self.prior)

    def _argmax_posterior(self, sample):
        
        """
        Given 1 sample, return the digit which maximizes the posteriori
        Inputs:
        sample (1D np.array): Pixel values of a single image
        """
        
        totalPosteriors = np.sum(norm.logpdf(np.tile(sample, (self.N_labels,1)),
            loc = self.mean, scale = self.std), axis = 1) + np.log(self.prior)
        return np.argmax(totalPosteriors)
                    
    def predict(self, data):

        """
        Predicting each image's label (digit) using maximum posteriori and naive bayes. 
        Inputs taken: test set (TsX) with all pixel values of each image 
        """

        totalPredictions = []
        for row in data:
            totalPredictions.append(self._argmax_posterior(row))
        return totalPredictions
  

def load_data(file_path):
    Numpyfile = scipy.io.loadmat(file_path) 
    trainX, trainY = Numpyfile['trX'], Numpyfile['trY']
    trainY = trainY.astype(np.int64)
    trainY = trainY.reshape([trainY.shape[1],])
    testX, testY = Numpyfile['tsX'], Numpyfile['tsY']
    testY = testY.astype(np.int64)
    testY = testY.reshape([testY.shape[1],])
    
    return trainX, trainY, testX, testY


# ------------- LOGISTIC REGRESSION -----------

class LogisticRegressionClassifier:
    def __init__(self, learning_rate=0.01, max_iterations=1000):

        '''
        Declare variables for LR class with learning rate, max iterations
        and likelihoods
        '''

        self.learning_rate  = learning_rate
        self.maximumIterations = max_iterations
        self.likelihoods    = []
        
        # Assume smallest value to epsilon to avoid log(0)
        self.epsilon = 1e-7

    def sigmoid(self, z):

        '''
        Calculate sigmoid function
        Inputs taken: numpy array of weighted samples
        '''

        sigmoid = (1/(1+np.exp(-z)))
        return sigmoid
    
    def mle(self, y_actual, y_predicted):

        '''
        Compute the maximum likelihood estimation / log-likelihood
        Inputs taken: Actual Y labels and Predicted Y labels
        '''

        y_predicted = np.maximum(np.full(y_predicted.shape, self.epsilon), np.minimum(np.full(y_predicted.shape, 1-self.epsilon), y_predicted))
        likelihood = (y_actual*np.log(y_predicted)+(1-y_actual)*np.log(1-y_predicted))     
        return np.mean(likelihood)
    
    def fit(self, X, y):

        '''
        Training the LR model by using gradient method algorithm and MLE
        Inputs taken: Train X samples (images) and Train Y (labels)
        '''
        
        # assume proper shape to X and initialize the weights
        self.weights = np.zeros((X.shape[1]))
        
        # Gradient Ascent
        for i in range(self.maximumIterations):
            # linear hypothesis
            z  = np.dot(X,self.weights)
          
            # Probabily value by computing sigmoid on z
            y_predicted = self.sigmoid(z)
            
            # Gradient values
            gradient = np.mean((y-y_predicted)*X.T, axis=1)
            
            # update weights where gradient is the ascent values
            self.weights +=  self.learning_rate*gradient
            
            # Log-likelihood / MLE
            likelihood = self.mle(y, y_predicted)

            self.likelihoods.append(likelihood)
    
    
    def predict_probability(self,X):

        '''
        Predicting sigmoid range probabilities with values between 0 and 1
        Inputs taken: Test samples (TsX)
        '''
               
        z = np.dot(X,self.weights)
        probabilities = self.sigmoid(z)
        
        return probabilities
    
    def predict(self, X, threshold=0.5):

        '''
        Classifying the test samples with appropriate digit labels
        Inputs taken: Test X images and threshold valye is 0.5. Value > 0.5 is 1, else 0
        '''

        # Thresholding probability to predict binary values
        LRpredictions = np.array(list(map(lambda x: 1 if x>threshold else 0, self.predict_probability(X))))    
        return LRpredictions

    def lr_accuracy(self, y_actual, y_predicted):

      '''
      Computing the accuracy of the LR model
      Inputs taken: Actual test Y labels, predicted Y labels
      '''

      count=0
      y_residual = y_predicted - y_actual
      count = np.count_nonzero(y_residual==0)   
      lr_accuracy = (count/len(y_predicted))*100
      return lr_accuracy

if __name__ == '__main__':

    trainX, trainY, testX, testY = load_data('mnist_data.mat')
    print("********** Results on MNIST Dataset (for Digits 7 and 8) **********")
    print("Data Specs: ")
    print("Training Data Size:", len(trainY))
    print("Testing Data Size:", len(testY))

    print("*****************************************")

    print("Training NB Model .... ")
    NBmodel = NaiveBayesClassifier(len(np.unique(trainY)))
    NBmodel.fit_classifier(trainX, trainY)

    pred_train = NBmodel.predict(trainX)
    pred_test = NBmodel.predict(testX)

    nb_accuracy = np.mean(trainY == pred_train)*100
    print("Naive Bayes Classifier Training Accuracy: ", nb_accuracy)

    acc_test = np.mean(testY == pred_test)*100
    print("Naive Bayes Classifier Testing Accuracy: ", acc_test)

    print("*****************************************")

    print("Training LR Model .... ")
    LRmodel = LogisticRegressionClassifier()
    LRmodel.fit(trainX,trainY)
    y_pred = LRmodel.predict(testX)
    print("Accuracy of LR: ", LRmodel.lr_accuracy(testY, y_pred))





