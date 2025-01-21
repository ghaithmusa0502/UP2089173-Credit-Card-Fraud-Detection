import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os
import json
import subprocess
import zipfile
import requests
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE

def yes_json_colab():
    uploaded = files.upload()  # Prompt user to upload kaggle.json
    # Check if the file is uploaded
    if 'kaggle.json' in uploaded:
        os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
        with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'wb') as f:
            f.write(uploaded['kaggle.json'])   
            os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/.kaggle')
            print("Kaggle authentication successful!")
            # Download and unzip the dataset
            os.system('kaggle datasets download -d mlg-ulb/creditcardfraud')
            os.system('unzip creditcardfraud.zip')
            os.remove("creditcardfraud.zip")  # Clean up the zip file
            # Load and return the dataset into a pandas DataFrame
            data = pd.read_csv('creditcard.csv')
            return data
    else:
        print("No kaggle.json file uploaded. Authentication failed.")

def upload_kaggle_json_not_colab():
    """
    Handles Kaggle authentication via file upload when not using Colab.
    """
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the Tkinter root window

        file_path = filedialog.askopenfilename(
            title="Select your Kaggle API JSON file",
            filetypes=[("JSON files", "*.json")]
        )

        if not file_path:
            raise FileNotFoundError("No file selected. Authentication failed.")

        with open(file_path, 'r') as file:
            kaggle_credentials = json.load(file)
        
        os.environ['KAGGLE_USERNAME'] = kaggle_credentials.get('username')
        os.environ['KAGGLE_KEY'] = kaggle_credentials.get('key')

        print("Kaggle authentication successful!")

        # Download dataset from Kaggle
        subprocess.run(['kaggle', 'datasets', 'download', '-d', 'mlg-ulb/creditcardfraud'])
        
        # Unzip the downloaded dataset
        with zipfile.ZipFile('creditcardfraud.zip', 'r') as zip_ref:
            zip_ref.extractall()
        
        os.remove('creditcardfraud.zip')
        
        # Read the CSV file into a DataFrame
        data = pd.read_csv('creditcard.csv')
        return data

    except Exception as e:
        print(f"Error: {e}")
        return None

def download_and_load_csv(url, file_path):
    """
    Download a CSV file from a URL and save it locally before loading it as a Pandas DataFrame.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            f.write(response.content)
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error downloading or loading file: {e}")
        return None

def preprocess_data(data, target_column):
    """
    Separate features and target, and standardize numerical features.
    """
    try:
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None, None


def preprocess_and_resample_data(data, target_column):
    """
    Standardize features and handle class imbalance using SMOTE.
    """
    try:
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
        return X_resampled, y_resampled
    except Exception as e:
        print(f"Error during preprocessing and resampling: {e}")
        return None, None


def build_neural_network(input_shape):
    """
    Build and compile a simple feedforward neural network.
    """
    try:
        model = Sequential([
            Input(shape=(input_shape,)),
            Dense(32, activation='relu'),
            Dropout(0.2),  # Add dropout for regularization
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    except Exception as e:
        print(f"Error while building the neural network: {e}")
        return None


def upload_csv():
    """
    Open a file dialog to select a CSV file and load it as a Pandas DataFrame.
    """
    try:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select a CSV file", 
            filetypes=[("CSV files", "*.csv")]
        )
        
        if not file_path:
            raise FileNotFoundError("No file selected.")
        
        df = pd.read_csv(file_path)
        print(f"Loaded file: {file_path}")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
