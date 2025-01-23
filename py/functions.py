import os
import json
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, f1_score
import subprocess
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
import zipfile
i = 0 
try: 
    from google.colab import files
except ModuleNotFoundError:
            i+=1
            pass
if i == 1:
    try: 
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
    except NameError:
            pass

# Function to create the kaggle.json file if it doesn't exist
def create_kaggle_file_prompt_Local():
    response = messagebox.askquestion("Create kaggle.json", "The kaggle.json file is missing. Would you like to create one?")
    return response

def ensure_kaggle_json_Local():
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")

    if os.path.exists(kaggle_json_path):
        print(f"Kaggle credentials found at {kaggle_json_path}.")
        return kaggle_json_path

    root = tk.Tk()
    root.withdraw()  # Hide the Tkinter root window

    # Ask if the user has a Kaggle API key
    has_kaggle_api = messagebox.askquestion("Kaggle API", "Do you have a Kaggle API key?")

    if has_kaggle_api == "yes":
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Open a file dialog to select kaggle.json
        kaggle_json_path = filedialog.askopenfilename(
            title="Select your kaggle.json file",
            filetypes=[("JSON Files", "*.json")]  # Restrict file type to JSON
        )
        if kaggle_json_path:
            print(f"Selected file: {kaggle_json_path}")
            kaggle_dir = os.path.expanduser("~/.kaggle")

            # Create the directory if it doesn't exist
            os.makedirs(kaggle_dir, exist_ok=True)

            # Move the selected file to the .kaggle directory
            shutil.move(kaggle_json_path, os.path.join(kaggle_dir, "kaggle.json"))
            print(f"Moved kaggle.json to {os.path.join(kaggle_dir, 'kaggle.json')}")

            # Set the environment variable
            os.environ['KAGGLE_CONFIG_DIR'] = kaggle_dir
            print(f"Set environment variable KAGGLE_CONFIG_DIR to {kaggle_dir}")
        else:
            print("No file selected. Authentication cannot proceed.")
            return None
    else:
        # If the user doesn't have the API key, create a default kaggle.json file with empty fields
        with open(kaggle_json_path, "w") as f:
            api_key = '{"username": "", "key": ""}'  # Empty values for public Kaggle key
            f.write(api_key)
        print(f"Created {kaggle_json_path} with empty credentials.")
        raise SystemExit("Please re-run the script.")
    return kaggle_json_path

# Function to authenticate with Kaggle API
def authenticate_kaggle_Local():
    try:
        api = KaggleApi()
        api.authenticate()
        print("Successfully authenticated with Kaggle API.")
        return api
    except kaggle.rest.ApiException as e:
        print("Authentication failed.")
        print("Error details:", e)
        print("You may need to update your Kaggle API token.")
        ensure_kaggle_json_Local()

# Function to download and extract Kaggle dataset
def Download_and_Extract_Kaggle_Dataset_Local():
    # Ensure Kaggle JSON file exists and authenticate
    kaggle_json_path = ensure_kaggle_json_Local()
    api = authenticate_kaggle_Local()

    # Specify dataset details
    dataset_owner = "mlg-ulb"
    dataset_name = "creditcardfraud"
    file_name = "creditcard.csv"

    try:
        print(f"Downloading {file_name} from Kaggle dataset {dataset_owner}/{dataset_name}...")
        api.dataset_download_file(dataset=f"{dataset_owner}/{dataset_name}",
                                  file_name=file_name,
                                  path=".")
        print(f"Downloaded {file_name} successfully!")

        # Extract the file if it's a zip file
        zip_path = f"{file_name}.zip"
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove(zip_path)
            print(f"Extracted {file_name} successfully!")
    except kaggle.rest.ApiException as e:
        print("Failed to download the file.")
        print("Error details:", e)


def preprocess_data(data, target_column):
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
    
def download_kaggle_dataset_colab():
    # Kaggle API file handling
    print("Do you have a Kaggle API file?")
    print("1. Yes")
    print("2. No")
    json_file = int(input("Enter the number of your choice: "))
    
    # Ensure .kaggle directory exists
    kaggle_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')

    if json_file == 1:
        # Upload Kaggle credentials
        print("Please upload your kaggle.json file")
        uploaded = files.upload()
        if 'kaggle.json' in uploaded:
            # Move uploaded file to correct location
            with open(kaggle_json_path, 'wb') as f:
                f.write(uploaded['kaggle.json'])
            
            # Set correct permissions
            os.chmod(kaggle_json_path, 0o600)
        else:
            print("No kaggle.json file uploaded.")
            return None
    else:
        # Create a placeholder kaggle.json
        with open(kaggle_json_path, 'w') as f:
            json.dump({"username": "", "key": ""}, f)
        
        print("Created placeholder kaggle.json. You'll need to manually add credentials.")

    # Set Kaggle config directory
    os.environ['KAGGLE_CONFIG_DIR'] = kaggle_dir

    try:
        # Download dataset
        subprocess.run(['kaggle', 'datasets', 'download', '-d', 'mlg-ulb/creditcardfraud'], check=True)
        
        # Unzip the dataset
        with zipfile.ZipFile('creditcardfraud.zip', 'r') as zip_ref:
            zip_ref.extractall()
        
        # Clean up zip file
        os.remove("creditcardfraud.zip")
        
        # Load and return the dataset
        data = pd.read_csv('creditcard.csv')
        return data
    except subprocess.CalledProcessError as e:
        print(f"Failed to download dataset: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

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
