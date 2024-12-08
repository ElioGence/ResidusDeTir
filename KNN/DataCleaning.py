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
        
