import requests
import pandas as pd
from google.colab import files
import tkinter as tk
from tkinter import filedialog


# Function to download and load the CSV file
def download_and_load_csv(url, filename):
    """
    Downloads a CSV file from the given URL and loads it into a pandas DataFrame.
    
    Parameters:
    url (str): The URL of the CSV file.
    filename (str): The name to save the file as.
    
    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    df = pd.read_csv(filename)
    return df

def upload_csv_google_colab():
    uploaded = files.upload()
    for filename, data in uploaded.items():
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(filename)
        # Display the first few rows of the DataFrame
        print(df.head())
        print(f"Filename: {filename}")
        break

def upload_csv_not_google_colab():
    # Create a Tkinter root window but don't display it
    root = tk.Tk()
    root.withdraw()
    
    # Open file dialog for selecting a CSV file
    file_path = filedialog.askopenfilename(
        title="Select a CSV file", 
        filetypes=[("CSV files", "*.csv")]
    )
    
    # If the user selected a file
    if file_path:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
        return df
    else:
        print("No file selected.")
        return None