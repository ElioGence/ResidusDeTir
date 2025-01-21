import os
import pandas as pd
import numpy as np
from FileManipulation import *

def cleaning_Slash(csv_path="datasets/training"):
    csv_files = [f for f in os.listdir(csv_path) if f.endswith('.csv')]
    
    for file in csv_files:
        csv_file = os.path.join(csv_path, file)
        df = pd.read_csv(csv_file, encoding='UTF-16', sep='\t')
        df['Sous-classe'] = df['Sous-classe'].replace("/", "", regex=True)
        save_Dataframe(df, file, os.path.join("results", "csv", "cleanedData"))
        
def cleaning_AVG(csv_path="datasets/training"):
    csv_files = [f for f in os.listdir(csv_path) if f.endswith('.csv')]
    
    for file in csv_files:
        csv_file = os.path.join(csv_path, file)
        df = pd.read_csv(csv_file, encoding='UTF-16', sep='\t')
       
        first_row = df.iloc[:1]
      
        df_rest = df.iloc[1:].copy()
        df_rest['Id'] = pd.to_numeric(df_rest['Id'], errors='coerce')  # Convert to numeric
        df_rest = df_rest.dropna(subset=['Id'])  # Drop rows where 'ID' is NaN
        
        # Combine the first row with the cleaned rest of the DataFrame
        df_cleaned = pd.concat([first_row, df_rest], ignore_index=True)
       
        save_Dataframe(df, file, os.path.join("results", "csv", "cleanedData"))
