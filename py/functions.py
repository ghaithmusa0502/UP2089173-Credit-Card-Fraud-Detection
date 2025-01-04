import requests
import pandas as pd

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
