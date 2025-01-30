import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
from sklearn.model_selection import RandomizedSearchCV

import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from ModelSetup import *
from FileManipulation import *
from DataCleaning import *

def trainingXGB(timeToken):
    Data = load_All_Data("datasets/cleanTraining")
    X, y = setup_X_y(timeToken, Data)
    le = LabelEncoder()
    y = le.fit_transform(y)

   
    print(np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    model = xgb.XGBClassifier(eta=0.06,max_depth=7,colsample_bytree=0.5,alpha=0.3,n_estimators=1000, n_jobs=-1) #(eta=0.07,max_depth=6,colsample_bytree=0.5,alpha=0.2,n_estimators=700) -> 95
    model.fit(X_train, y_train)
    p_model = model.predict(X_test)

    s_model = model.score(X_test, y_test)

    if not os.path.exists("XGB/confusion_matrix.csv"):
        cm = confusion_matrix(y_test, p_model)
        dispcm = ConfusionMatrixDisplay(confusion_matrix=cm)
        df = pd.DataFrame(cm)
        # Sauvegarder la matrice dans un fichier en format pouvant être chargé 
        df.to_csv("XGB/confusion_matrix.csv")
        dispcm.plot()
        plt.show()

    print(s_model)
    
def XGB_find_best_params(timeToken):
    Data = load_All_Data("datasets/training")
    X, y = setup_X_y(timeToken, Data)
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = [label - 1 if label >= 50 else label for label in y]
    y = [label - 1 if label >= 50 else label for label in y]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    param_dist = {
        'eta': [0.01, 0.03, 0.05, 0.07, 0.1],
        'max_depth': [3, 5, 7, 9],
        'colsample_bytree': [0.3, 0.5, 0.7],
        'alpha': [0.1, 0.3, 0.5],
        'n_estimators': [100, 300, 500, 700, 1000]
    }

    # Configurer le modèle
    model = xgb.XGBClassifier(n_jobs=-1)

    # RandomizedSearchCV avec 10 essais
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,  # Nombre d'essais
        scoring='accuracy',
        cv=3,  # Validation croisée
        verbose=1,
        random_state=42
    )

    # Exécuter la recherche
    random_search.fit(X_train, y_train)

    # Afficher les meilleurs paramètres
    print(f"Best parameters found: {random_search.best_params_}")
    print(f"Best accuracy (cross-validation): {random_search.best_score_:.4f}")

    # Tester le meilleur modèle
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test set accuracy with best parameters: {accuracy:.4f}")

def XGB_learning_curve(timeToken):
    Data = load_All_Data("datasets/cleanTraining")
    X, y = setup_X_y(timeToken, Data)
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Compter les occurrences de chaque classe
    class_counts = Counter(y)
    
    # Filtrer les classes avec moins de 2 occurrences
    valid_classes = [cls for cls, count in class_counts.items() if count >= 2]
    mask = np.isin(y, valid_classes)
    X, y = X[mask], y[mask]

    # Réindexer les classes pour qu'elles soient continues
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=5,stratify=y)
    print(np.unique(y))

    model = xgb.XGBClassifier(eta=0.05, subsample=0.5,colsample_bytree=0.5,alpha=0.3,n_estimators=700, n_jobs=-1, eval_metric='mlogloss') #(eta=0.07,max_depth=6,colsample_bytree=0.5,alpha=0.2,n_estimators=700) -> 95
    evalset = [(X_train, y_train), (X_test, y_test)]
    print("y_train:", np.unique(evalset[0][1]))
    print("--------------------")
    print("y_test:", np.unique(evalset[1][1]))

    model.fit(X_train, y_train, eval_set=evalset)
    p_model = model.predict(X_test)

    s_model = accuracy_score(y_test, p_model)


    results = model.evals_result()

    plt.plot(results['validation_0']['mlogloss'], label='train')
    plt.plot(results['validation_1']['mlogloss'], label='test')
    save_Plot(timeToken,"XGB/XGB_learning_curve")
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    
def analyse_confusion_matrix(timeToken):
    cm = pd.read_csv("XGB/confusion_matrix.csv", index_col=0).values
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    class_counts = cm.sum(axis=1)
    for idx, (accuracy, count) in enumerate(zip(class_accuracy, class_counts)):
        print(f"Class {idx} accuracy: {accuracy:.2f}, Nb Valeurs: {count}")

    results = pd.DataFrame({
        'Class': range(len(class_accuracy)),
        'Accuracy': class_accuracy,
        'Count': class_counts
    })
    results.to_csv("XGB/class_accuracy.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.scatter(class_counts, class_accuracy)
    plt.xlabel('Nombre de valeurs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Nombre de valeurs par classe')
    #save_Plot(timeToken,"XGB/XGB_Class_Accuracy")

    plt.show()

def analyze_class_acurracy():
    results = pd.read_csv("XGB/class_accuracy.csv")
    filtered_results = results[results['Count'] > 8000]
    mean_accuracy = filtered_results['Accuracy'].mean()
    print(f"Précision moyenne pour les classes avec plus de 1000 valeurs: {mean_accuracy:.4f}")

def tSNEXGB(timeToken):
    Data = load_All_Data("datasets/cleanTraining")
    X, y = setup_X_y(timeToken, Data)
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    model = TSNE(perplexity=50, learning_rate=500, max_iter=1000, init='pca', n_components=2, random_state=42, n_jobs=-1)
    embedding = model.fit_transform(X)  # X is your combined dataset (train + test)
    
    # Split the embedding manually into training and test sets
    X_train_embedded = model.fit_transform(X_train)
    X_test_embedded = model.fit_transform(X_test)

    # Step 2: Train an XGBoost classifier on the t-SNE-reduced training data
    model = xgb.XGBClassifier(eta=0.06,max_depth=7,colsample_bytree=0.5,alpha=0.3,n_estimators=1000, n_jobs=-1)
    model.fit(X_train_embedded, y_train)

    # Step 3: Test the XGBoost classifier on the t-SNE-reduced test data
    y_pred = model.predict(X_test_embedded)

    # Step 4: Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    if not os.path.exists("XGB/confusion_matrix_tSNE.csv"):
        cm = confusion_matrix(y_test, y_pred)
        dispcm = ConfusionMatrixDisplay(confusion_matrix=cm)
        df = pd.DataFrame(cm)
        # Sauvegarder la matrice dans un fichier en format pouvant être chargé 
        df.to_csv("XGB/confusion_matrix_tSNE.csv")
        dispcm.plot()
        plt.show()



currentDT = datetime.datetime.now()
timeToken = "("+str(currentDT.month)+str(currentDT.day)+str(currentDT.hour)+str(currentDT.minute)+str(currentDT.second)+")_"
print("Time Token : "+timeToken)


#trainingXGB(timeToken)
#analyse_confusion_matrix(timeToken)
#analyze_class_acurracy()
#XGB_find_best_params(timeToken)
XGB_learning_curve(timeToken)
#tSNEXGB(timeToken)