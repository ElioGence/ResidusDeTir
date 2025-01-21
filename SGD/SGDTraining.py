from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve

import datetime
import random
from random import randrange
import pandas as pd
import matplotlib.pyplot as plt

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from ModelSetup import *
from FileManipulation import *
from DataCleaning import *

import warnings #Should be avoided but we have too few data in certain categories and we force cv=5
warnings.filterwarnings("ignore", category=UserWarning)

def get_Best_Parameters_SGD(X, y):
    param_grid = {
        'alpha': [0.0001, 0.001, 0.01],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'learning_rate': ['constant', 'adaptive'],
        'eta0': [0.01, 0.1, 1],
        'max_iter': [1000, 2000]
    }
    grid_search = GridSearchCV(SGDClassifier(), param_grid, cv=5)
    grid_search.fit(X, y)
    print(grid_search.best_params_)

def save_Learning_Curve(timeToken, model, X_train, y_train):
    """
    Function to save the learning curve
    """
    N, train_score, val_score = learning_curve(model, X_train, y_train,
                                           train_sizes=np.linspace(0.1, 1, 10), cv=5)

    plt.figure(figsize=(8, 5))
    plt.plot(N, train_score.mean(axis=1), label='train')
    plt.plot(N, val_score.mean(axis=1), label='validation')
    plt.xlabel('train_sizes')
    plt.legend()
    save_Plot(timeToken,"SGD_Learning_Curve.png")
    plt.show()


def training(timeToken):
    Data = load_All_Data("datasets/training")
    
    save_SubClass_Count(timeToken, Data)
    
    X, y = setup_X_y(timeToken, Data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    
    #get_Best_Parameters_SGD(X_train, y_train)
    
    #model = SGDClassifier(alpha= 0.0001, eta0= 1, learning_rate='adaptive',max_iter=1000, penalty='l1')
    
    #save_Validation_Scores(timeToken, model, X_train, y_train)
    
    #save_Learning_Curve(timeToken, model, X_train, y_train)
    
    #model.fit(X_train, y_train)

    #predictedDF=setup_Prediction(model,X_train) 
    """ UPDATE TO MAKE A PREDICTION ON THE TEST VALUES"""
    
    #save_Prediction(timeToken, predictedDF, "Result_SGD.csv", Data)

currentDT = datetime.datetime.now()
timeToken = "("+str(currentDT.month)+str(currentDT.day)+str(currentDT.hour)+str(currentDT.minute)+str(currentDT.second)+")_"
print("Time Token : "+timeToken)

training(timeToken)














