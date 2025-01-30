import datetime
import pandas as pd
import matplotlib.pyplot as plt
from ModelSetup import *
from FileManipulation import *
import seaborn as sns
from DataCleaning import *
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC

def trainingRDF(timeToken):
    Data = load_All_Data("datasets/training")
    X, y = setup_X_y(timeToken, Data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    model = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1)

    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))

def get_best_params_RDF():
    Data = load_All_Data("datasets/training")
    X, y = setup_X_y(timeToken, Data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    param_grid = {
        'n_estimators': [200, 300, 500],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)

def RDF_validation_curve():
    Data = load_All_Data("datasets/training")
    X, y = setup_X_y(timeToken, Data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    k = [100, 200, 300, 400, 500, 700, 1000]

    train_scores, val_scores = validation_curve(RandomForestClassifier(), X_train, y_train,
                                                param_name='n_estimators', param_range=k, cv=5, n_jobs=-1)

    plt.figure(figsize=(8, 5))
    plt.plot(k, val_scores.mean(axis=1), label='Validation Score')
    plt.plot(k, train_scores.mean(axis=1), label='Train Score')
    plt.ylabel('Score')
    plt.xlabel('Number of Estimators')
    plt.title("Validation Curve for Random Forest")
    plt.legend()
    plt.grid()
    save_Plot(timeToken,"RDF_Validation_Scores.png")
    plt.show()

def RDF_matrice_confusion():
    Data = load_All_Data("datasets/training")
    X, y = setup_X_y(timeToken, Data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    model = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels = model.classes_)
    dispcm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    print(confusion_matrix)

    df = pd.DataFrame(cm)
    # Sauvegarder la matrice dans un fichier en format pouvant être chargé 
    df.to_csv("RDF/confusion_matrix.csv")

    dispcm.plot()
    plt.show()

    #plt.figure(figsize=(8, 6))
    #plt.title("Matrice de confusion pour le modèle Random Forest")
    #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    #save_Plot(timeToken,"RDF_Confusion_Matrix.png")
    #plt.show()

def analyze_matrix():
    cm = pd.read_csv("RDF/confusion_matrix.csv", index_col=0).values
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    class_counts = cm.sum(axis=1)
    for idx, (accuracy, count) in enumerate(zip(class_accuracy, class_counts)):
        print(f"Classe {idx} accuracy: {accuracy:.2f}, Nb Valeurs: {count}")

    plt.figure(figsize=(8, 5))
    plt.scatter(class_counts, class_accuracy)
    plt.xlabel('Nombre de valeurs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Nombre de valeurs par classe')
    plt.grid()
    plt.show()
    results = pd.DataFrame({
        'Classe': range(len(class_accuracy)),
        'Accuracy': class_accuracy,
        'Nombre': class_counts
    })
    results.to_csv("RDF/class_accuracy.csv", index=False)

currentDT = datetime.datetime.now()
timeToken = "("+str(currentDT.month)+str(currentDT.day)+str(currentDT.hour)+str(currentDT.minute)+str(currentDT.second)+")_"
print("Time Token : "+timeToken)

RDF_matrice_confusion()
analyze_matrix()