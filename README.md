# UP2089173-Credit-Card-Fraud-Detection

This repository contains code for detecting fraudulent credit card transactions using machine learning techniques. The dataset used consists of anonymised credit card transaction details, where the target variable (Class) indicates whether a transaction was fraudulent (1) or non-fraudulent (0). The project includes various methods such as standard neural networks, handling imbalanced data with SMOTE, and performance evaluation using multiple metrics and visualisations.

This dataset can be found at https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download and downloaded by you or let this can choose download this repository and download the code that way.

Due to the large file size of this dataset downloading the dataset in the conventional manner will not be sufficient, you will have to clone it. If you do not want to clone it, then you can go through any of the .ipynb files and download it through that.

In Kaggle API's there is a requirement of a .json file is needed to access and download the dataset. However when writing this code I have found that any .json file that has a username and key written in the file will be enough. So when going through one of the routes and it asks for a kaggle.json file entering no will grant you access and download the file in a similar in the manner if you had one. 

If you do not have a Kaggle API and wish to run this on your local complier it will ask you to run this code twice, that is normal and after it creates the json file it will run as normal.

To run this code you will need python 3.11.9 and these libraries:

1) OS
2) zipfile: Built-in
3) importlib.util: Built-in 
4) tkinter: Built-in 
5) JSON: Built-in 
6) google.colab: Built-in 
7) Pandas
8) Matplotlib
9) Seaborn
10) Scikit-learn
11) TensorFlow
12) Imbalanced-learn
13) Requests
14) Kaggle
15) Kagglehub

You can install them using this on your python notebook:
pip install pandas scikit-learn tensorflow matplotlib seaborn imbalanced-learn requests kaggle kagglehub
