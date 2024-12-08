from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
import datetime
import random
from random import randrange
import pandas as pd
import matplotlib.pyplot as plt
from ModelSetup import *
from FileManipulation import *
from DataCleaning import *

import warnings #Should be avoided but we have too few data in certain categories and we force cv=5
warnings.filterwarnings("ignore", category=UserWarning)

"""
Need to update to avoid overfitting here some possible measures :
Distance Weights (custom)
Distance Metric 
More Data (some categories are under represented)

Once we have enough data we can also use StratifiedKFold instead of regular cv fold
"""

def get_Best_Number_Neighbors(timeToken,X,y,kmax=30):
    """
    Return the best number of neighbors for X and y.
    Also save the plot and return best_k and best_score
    """
    scores_uniform = []
    best_k_uniform = 1
    best_score_uniform = 0
    
    for k in range(best_k_uniform, kmax):
        # Creation of the model
        model = KNeighborsClassifier(n_neighbors=k)        
            
        min_class_count = min(Counter(y).values()) #To avoid using a cv too big
        cv_folds = min(5, min_class_count)  
        if cv_folds < 2:
            cv_folds = 5
        
        cv_scores = cross_val_score(model, X, y, cv=cv_folds)
        mean_score = np.mean(cv_scores)
        scores_uniform.append(mean_score)
        
        if mean_score > best_score_uniform:
           best_k_uniform = k
           best_score_uniform = mean_score
        
    print(f"UNIFORM = Best k: {best_k_uniform}, Best Score: {best_score_uniform:.4f}")
   
    scores_distance = []
    best_k_distance = 1
    best_score_distance = 0
    
    for k in range(best_k_distance, kmax):
        # Creation of the model
        model = KNeighborsClassifier(n_neighbors=k)        
            
        min_class_count = min(Counter(y).values()) #To avoid using a cv too big
        cv_folds = min(5, min_class_count)  
        if cv_folds < 2:
            cv_folds = 5
        
        cv_scores = cross_val_score(model, X, y, cv=cv_folds)
        mean_score = np.mean(cv_scores)
        scores_distance.append(mean_score)
        
        if mean_score > best_score_distance:
           best_k_distance = k
           best_score_distance = mean_score
        
    print(f"DISTANCE = Best k: {best_k_distance}, Best Score: {best_score_distance:.4f}")
   
    plt.figure(figsize=(12, 4))
    plt.suptitle("Accuracy comparaison Weights")
    
    plt.subplot(121)
    plt.plot(range(1, kmax), scores_uniform, marker='o', linestyle='-')
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Accuracy")
    plt.title('Uniform')
    
    plt.subplot(122)
    plt.plot(range(1, kmax), scores_distance, marker='o', linestyle='-')
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Accuracy")
    plt.title('Distance')
    
    save_Plot(timeToken+"KNN_Weight_Comparaison.png")
    plt.show()
    
    best_k=best_k_distance
    best_score=best_score_distance
    if best_score_uniform>best_score_distance:
        best_k=best_k_uniform
        best_score=best_score_uniform
    return best_k, best_score

def save_Validation_Scores(timeToken, model, X_train, y_train, kmax=20):
    """
    Function to save the scores for train and validation (only use this if you want to save the validation curve otherwise redundant)
    """
    k = np.arange(1, kmax)

    train_scores, val_scores = validation_curve(model, X_train, y_train,
                                              param_name='n_neighbors', param_range=k, cv=5)
    
    plt.figure(figsize=(8, 5))
    plt.plot(k, val_scores.mean(axis=1), label='Validation Score')
    plt.plot(k, train_scores.mean(axis=1), label='Train Score')
    plt.ylabel('Score')
    plt.xlabel('Number of Neighbors (k)')
    plt.title("Validation Curve")
    plt.legend()
    plt.grid()
    save_Plot(timeToken+"KNN_Validation_Scores.png")
    plt.show()

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
    save_Plot(timeToken+"KNN_Learning_Curve.png")
    plt.show()


def training(timeToken):
    Data = load_All_Data("datasets/training")
    
    #save_SubClass_Count(timeToken, Data)
    
    X, y = setup_X_y(timeToken, Data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    best_k, best_score=get_Best_Number_Neighbors(timeToken, X_train,y_train,30)
    
    model = KNeighborsClassifier(n_neighbors=best_k)
    
    #save_Validation_Scores(timeToken, model, X_train, y_train)
    
    #save_Learning_Curve(timeToken, model, X_train, y_train)
    
    model.fit(X_train, y_train)

    #predictedDF=setup_Prediction(model,X_train) 
    """ UPDATE TO MAKE A PREDICTION ON THE TEST VALUES"""
    
    #save_Prediction(timeToken, predictedDF, "Result_KNN.csv", Data)
    """ Last function need to be changed so that the differences can be visualised instead of just saving a csv"""

currentDT = datetime.datetime.now()
timeToken = "("+str(currentDT.month)+str(currentDT.day)+str(currentDT.hour)+str(currentDT.minute)+str(currentDT.second)+")_"
print("Time Token : "+timeToken)

#cleaning_Slash()
training(timeToken)














