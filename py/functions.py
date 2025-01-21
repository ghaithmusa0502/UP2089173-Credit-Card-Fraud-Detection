import tkinter as tk
from tkinter import filedialog
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE

def upload_csv():
    """
    Open a file dialog to select a CSV file and load it as a Pandas DataFrame.
    """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a CSV file", 
        filetypes=[("CSV files", "*.csv")]
    )
    if file_path:
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded file: {file_path}")
            return df
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    else:
        print("No file selected")
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
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dropout(0.2),  # Add dropout for regularization
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model
