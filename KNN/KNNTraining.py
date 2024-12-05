from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
import pandas as pd
import matplotlib.pyplot as plt
from ModelSetup import *
from FileManipulation import *

def get_Best_Number_Neighbors(X,y,kmax=30):
    """
    Return the best number of neighbors for X and y.
    Also save the plot and return best_k and best_score
    """
    scores = []
    best_k = 1
    best_score = 0
    
    for k in range(best_k, kmax):
        # Creation of the model
        model = KNeighborsClassifier(n_neighbors=k)        
            
        min_class_count = min(Counter(y).values()) #To avoid using a cv too big
        cv_folds = min(5, min_class_count)  
        if cv_folds < 2:
            cv_folds = 5
        
        cv_scores = cross_val_score(model, X, y, cv=cv_folds)
        mean_score = np.mean(cv_scores)
        scores.append(mean_score)
        
        if mean_score > best_score:
           best_k = k
           best_score = mean_score
        
    print(f"Best k: {best_k}, Best Score: {best_score:.4f}")
   
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, kmax), scores, marker='o', linestyle='-')
    plt.title("Cross-Validation Accuracy")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Mean Cross-Validation Accuracy")
    plt.grid()
    save_Plot("Cross_Validation.png")
    plt.show()
    
    return best_k, best_score

def save_Scores(model, X_train, X_test, y_train, y_test, kmax=20):
    """
    Function to save the scores for train and test (only use this if you want to save the validation curve)
    
    ! NEED TO COMPARE MORE THAN JUST KNN, LIKE TRAIN SIZE FOR EXAMPLE !
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
    save_Plot("Scores.png")
    plt.show()


Data = load_All_Data("datasets/training")

X, y = setup_X_y(Data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
best_k, best_score=get_Best_Number_Neighbors(X_train,y_train,30)

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)

save_Scores(model, X_train, X_test, y_train, y_test)

predictedDF=setup_Prediction(model,X_train)

save_Prediction(predictedDF, "Result_KNN.csv", Data)
""" Last line need to be changed so that the differences can be visualised instead of just saving a csv"""
