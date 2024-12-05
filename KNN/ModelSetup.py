import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from FileManipulation import *
from collections import Counter
import io

def save_SubClass_Count(Data=None):
    """
    Function save how many of each subclass there is in the data given 
    If no data specified it will take by default the one in datasets/training
    """
    if Data is None:
        Data = load_All_Data("datasets/training");
    sous_classe_count =Data.loc[:, (Data.columns == "Sous-classe")]
    sous_classe_count=sous_classe_count['Sous-classe'].value_counts()
    sous_classe_count.column = ['Sous-classe','Nombre de valeur']
    sous_classe_count.to_csv(r"results\csv\Info_SousClasseDecompte.csv", sep=';', index=True)

def setup_X_y(Data=None):
    """
    Function to setup the X and y for Machine Learning
    If no data specified it will take by default the one in datasets/training
    """
    #Loading and cleaning data
    if Data is None:
        Data = load_All_Data("datasets/training");
    #We only keep the ratio and "sous_classe" which will be the target (y) for our model
    gunshot_data = Data.loc[:, Data.columns.str.contains(" ratio")| (Data.columns == "Sous-classe")]
    gunshot_data = gunshot_data.replace(',', '.', regex=True).apply(pd.to_numeric, errors='ignore')    
        
    
    # Factorize the column (string to numerical value)
    gunshot_data['ValueKNN'] = pd.factorize(gunshot_data['Sous-classe'])[0]
    
    #We save in a csv the ValueKNN for each sous-classe 
    sousClasseCSV = gunshot_data[['ValueKNN', 'Sous-classe']]
    sousClasseCSV = sousClasseCSV.drop_duplicates(subset='Sous-classe', keep='first')
    sousClasseCSV.to_csv(r"results\csv\Info_CorrespondanceSousClasse.csv", sep=';', columns=['ValueKNN', 'Sous-classe'], index=False)
    
    y = gunshot_data['ValueKNN']
    
    X = gunshot_data.drop('Sous-classe', axis=1)    
    X = X.drop('ValueKNN', axis=1)   
    #We fill with 0 the non existant value
    X = X.fillna(0)
    return X, y

def save_Train_Plot(X_train, X_test, y_train, y_test, filename="TrainingPlot.png"):
    x_min = min(X_train.iloc[:, 0].min(), X_test.iloc[:, 0].min())
    x_max = max(X_train.iloc[:, 0].max(), X_test.iloc[:, 0].max())
    y_min = min(X_train.iloc[:, 1].min(), X_test.iloc[:, 1].min())
    y_max = max(X_train.iloc[:, 1].max(), X_test.iloc[:, 1].max())
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(121)
    plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, alpha=0.8)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title('Train set')
    
    plt.subplot(122)
    plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, alpha=0.8)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title('Test set')
    
    save_Plot(filename)

def setup_Prediction(model,X,Data=None):
    """
    Function to do a prediction
    If no data specified it will take by default the one in datasets/predict
    """
    #Now we check the predicting
    if Data is None:
        Data = load_All_Data("datasets/predict")
    Data = Data.loc[:, Data.columns.str.contains(" ratio")]
    Data = Data.replace(',', '.', regex=True).apply(pd.to_numeric, errors='ignore')
    Data = Data.fillna(0)
    
    #We make sure all columns we had in training are still there (otherwise we get an error)
    missing_cols = set(X.columns) - set(Data.columns)
    for col in missing_cols:
        Data[col] = 0
        
    #Align column order
    Data = Data[X.columns]

    predictedData = model.predict(Data)
    
    predictedDF = pd.DataFrame(predictedData, columns=['Predicted-SubClass'])
    
    return predictedDF
    
def save_Prediction(predictedDF, filename, Data=None):   
    """
    Function to save the prediction in a CSV 
    
    ! NEED TO BE UPDATED TO ALSO DO THE COMPARAISON BEETWEEN THE PREDICTION AND THE EXPECTED INSTEAD OF DOING IT IN EXCEL !
    
    If no data specified it will take by default the one in datasets/training
    """
    if Data is None:
        Data = load_All_Data("datasets/training")
    sous_classe_column = Data.loc[:, (Data.columns == "Sous-classe")]
    gunshot_data = Data.loc[:, (Data.columns == "Sous-classe")]
    gunshot_data = gunshot_data.replace(',', '.', regex=True).apply(pd.to_numeric, errors='ignore')    
    gunshot_data['ValueKNN'] = pd.factorize(gunshot_data['Sous-classe'])[0]
    
    #We turn back the numerical value into string
    categories = pd.factorize(gunshot_data['Sous-classe'])[1]
    predictedDF['Predicted-SubClass'] = predictedDF['Predicted-SubClass'].map(lambda x: categories[x % len(categories)])
    
    
    # Concatenate the predictions with the sous class expected
    Result = pd.concat([sous_classe_column, predictedDF], axis=1)
    
    #Finally we save the prediction
    save_Dataframe(Result, filename)



