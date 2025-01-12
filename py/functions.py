import tkinter as tk
from tkinter import filedialog
import pandas as pd
import requests

def upload_csv_not_google_colab():
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select a CSV file", 
        filetypes=[("CSV files", "*.csv")]
    )
    
    if file_path:
        df = pd.read_csv(file_path)
        print(f"Loaded file: {file_path}")
        return df
    else:
        print("No file selected")
        return None

def upload_csv_google_colab():
    from google.colab import files
    uploaded = files.upload()
    first_file = next(iter(uploaded))
    df = pd.read_csv(first_file)
    return df

def download_and_load_csv(url, file_path):
    response = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(response.content)
    return pd.read_csv(file_path)