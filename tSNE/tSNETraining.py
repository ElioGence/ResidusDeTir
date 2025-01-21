from sklearn.manifold import TSNE
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


import datetime
import random
from random import randrange
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from ModelSetup import *
from FileManipulation import *
from DataCleaning import *

import warnings #Should be avoided but we have too few data in certain categories and we force cv=5
warnings.filterwarnings("ignore", category=UserWarning)

def get_Best_Parameters_tSNE(X, y):
    param_grid = {
        'perplexity': [5, 30, 50],         # Typical range
        'learning_rate': [10, 200, 500],   # Typical range
        'max_iter': [1000],                # Set to a fixed value for simplicity
        'init': ['pca'],                   # Default initialization
        'n_components': [2],               # Dimensionality reduction to 2D
    }
    grid = ParameterGrid(param_grid)
    
    results = []
    
    for params in grid:
        print(f"Testing with parameters: {params}")
        tsne = TSNE(
            perplexity=params['perplexity'],
            learning_rate=params['learning_rate'],
            n_iter=params['max_iter'],
            init=params['init'],
            n_components=params['n_components'],
            random_state=42
        )
    
    embedding = tsne.fit_transform(X)
    
    distances = pairwise_distances(X)
    stress = np.sum((distances - pairwise_distances(embedding)) ** 2)
    results.append({'params': params, 'stress': stress})

    return results

def randomForest_tSNE(X, X_train, X_test, y_train, y_test):
    model = TSNE( perplexity=50, learning_rate=500, max_iter=1000, init='pca', n_components=2, random_state=42)
    embedding = model.fit_transform(X)  # X is your combined dataset (train + test)
    
    # Split the embedding manually into training and test sets
    #embedding_train = embedding[:len(X_train)]
    #embedding_test = embedding[len(X_train):]
    
    X_train_embedded = model.fit_transform(X_train)
    X_test_embedded = model.fit_transform(X_test)
    
    # Train a classifier on the t-SNE-reduced data
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_embedded, y_train)
    
    # Test the classifier on the t-SNE-reduced test data
    y_pred = clf.predict(X_test_embedded)
    
    # Evaluate accuracy
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
def xgBoost_tSNE(X, X_train, X_test, y_train, y_test):
    # Step 1: Apply t-SNE to reduce dimensionality of the dataset
    model = TSNE(perplexity=50, learning_rate=500, max_iter=1000, init='pca', n_components=2, random_state=42)
    embedding = model.fit_transform(X)  # X is your combined dataset (train + test)
    
    # Split the embedding manually into training and test sets
    X_train_embedded = model.fit_transform(X_train)
    X_test_embedded = model.fit_transform(X_test)
    
    # Step 2: Train an XGBoost classifier on the t-SNE-reduced training data
    clf = XGBClassifier(n_jobs=10, use_label_encoder=False, eval_metric='logloss', random_state=42)
    clf.fit(X_train_embedded, y_train)
    
    # Step 3: Test the XGBoost classifier on the t-SNE-reduced test data
    y_pred = clf.predict(X_test_embedded)
    
    # Step 4: Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    return clf, accuracy
    
def training(timeToken):
    Data = load_All_Data("datasets/training")
    
    save_SubClass_Count(timeToken, Data)
    
    X, y = setup_X_y(timeToken, Data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    
    #results = get_Best_Parameters_tSNE(X_train, y_train)
    #sorted_results = sorted(results, key=lambda x: x['stress'])
    #print("Best parameters:", sorted_results[0])
    
    xgBoost_tSNE(X, X_train, X_test, y_train, y_test)(X, X_train, X_test, y_train, y_test)

    #save_Prediction(timeToken, predictedDF, "Result_tSNE.csv", Data)

currentDT = datetime.datetime.now()
timeToken = "("+str(currentDT.month)+str(currentDT.day)+str(currentDT.hour)+str(currentDT.minute)+str(currentDT.second)+")_"
print("Time Token : "+timeToken)

training(timeToken)

#logical_cores = os.cpu_count()
#print(f"Logical Cores: {logical_cores}")













