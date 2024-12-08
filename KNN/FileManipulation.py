import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_Data(file):
    """
    Load a single file from datasets
    """
    csv_path = os.path.join("datasets", file)
    return pd.read_csv(csv_path, encoding='UTF-16', sep='\t')


def load_All_Data(csv_path="datasets"):    
    """
    Load all files from a path
    """
    csv_files = [f for f in os.listdir(csv_path) if f.endswith('.csv')]
    
    dataframes = []
    
    for file in csv_files:
        csv_file = os.path.join(csv_path, file)
        df = pd.read_csv(csv_file, encoding='UTF-16', sep='\t', low_memory=False) #Not sure for the low_memory
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def save_Plot(filename, directory="results/plot", dpi=300):
    """
    Save a plot without overwriting existing files
    
    Parameters:
        dpi (int): Resolution of the saved image.
    """
    filename=timeToken+filename
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Split filename into name and extension
    base, ext = os.path.splitext(filename)
    filepath = os.path.join(directory, filename)
    
    # Add a unique suffix if the file exists
    counter = 1
    while os.path.exists(filepath):
        filepath = os.path.join(directory, f"{base}_{counter}{ext}")
        counter += 1

    # Save the plot
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved as: {filepath}")
    
    
def save_Dataframe(timeToken, df, filename, directory="results/csv"):
    """
    Save a DataFrame as a CSV file without overwriting existing files
    """
    filename=timeToken+filename
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Split filename into name and extension
    base, ext = os.path.splitext(filename)
    filepath = os.path.join(directory, filename)
    
    # Add a unique suffix if the file exists
    counter = 1
    while os.path.exists(filepath):
        filepath = os.path.join(directory, f"{base}_{counter}{ext}")
        counter += 1

    # Save the DataFrame
    df.to_csv(filepath, sep='\t', index=True, encoding='utf-16')
    print(f"DataFrame saved as: {filepath}")
    
    
def save_Dataframe_Info(timeToken, df, filename="Info_Gunshotdata.csv", directory="results/csv"):
    """
    Used to save .info of a dataframe
    Be carefull, it will overwrite the file
    """
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    filename=timeToken+filename
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w') as f:
        f.write(df)
    
