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
from google.colab import files  # Ensure this is imported at the very top
import subprocess


def yes_json_colab():
    uploaded = files.upload()  # Prompt user to upload kaggle.json
    # Check if the file is uploaded
    if 'kaggle.json' in uploaded:
        os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
        with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'wb') as f:
            f.write(uploaded['kaggle.json'])
        os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/.kaggle')
        print("Kaggle authentication successful!")
        
        # Download dataset using subprocess
        subprocess.run(['kaggle', 'datasets', 'download', '-d', 'mlg-ulb/creditcardfraud'])
        
        # Unzip the dataset
        with zipfile.ZipFile('creditcardfraud.zip', 'r') as zip_ref:
            zip_ref.extractall()
        
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
        # Separate features and target variable
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        # Standardize the features to have zero mean and unit variance.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Return scaled features and target
        return X_scaled, y
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None, None


def preprocess_and_resample_data(data, target_column):
    try:
        # Separate features and target variable
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        # Standardize the features to have zero mean and unit variance.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Apply SMOTE  to oversample the minority class 
        # It does this by generating synthetic samples based on the feature space of existing minority-class data.
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
        # Return resampled features and target
        return X_resampled, y_resampled
    except Exception as e:
        # Handle any errors that occur during preprocessing
        print(f"Error during preprocessing and resampling: {e}")
        return None, None

def build_neural_network(input_shape):
    try:
        # Create a Sequential model, which allows stacking layers one by one
        model = Sequential([
            # Input layer: Defines the input shape for the model
            # 'input_shape' specifies the number of features in the input data
            Input(shape=(input_shape,)),
            # First hidden layer: Fully connected (Dense) layer with 32 neurons and ReLU activation
            # ReLU (Rectified Linear Unit) activation helps in introducing non-linearity, making the model capable of learning complex patterns
            Dense(32, activation='relu'),
            # Dropout layer: Randomly sets 20% of the neurons to zero during training
            # Helps prevent overfitting by reducing reliance on specific neurons
            Dropout(0.2),
            # Second hidden layer: Another Dense layer with 16 neurons and ReLU activation
            # Further processes the features learned from the first hidden layer
            Dense(16, activation='relu'),
            # Output layer: Single neuron with sigmoid activation for binary classification
            # Sigmoid activation squashes the output to a range between 0 and 1, representing probabilities
            Dense(1, activation='sigmoid')
        ])
        # Compile the model: Specify optimizer, loss function, and evaluation metrics
        model.compile(
            # Adam optimizer: Adaptive optimization algorithm that adjusts the learning rate during training
            # Well-suited for most deep learning tasks
            optimizer=Adam(learning_rate=0.001),
            # Loss function: Binary cross-entropy for binary classification
            # Measures the difference between predicted probabilities and actual class labels
            loss='binary_crossentropy',
            # Metrics: Accuracy to evaluate the percentage of correct predictions during training
            metrics=['accuracy']
        )
        # Return the compiled model, ready for training
        return model

    except Exception as e:
        # Exception handling: Print the error message if any part of model creation fails
        print(f"Error while building the neural network: {e}")
        # Return None to indicate failure to build the model
        return None


def upload_csv():
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
