import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

def load_wine_white(path="./", missing_strategy="drop"):
    
    # white wine dataset
    data = pd.read_csv(path + "uci_regression/wine_white/wine_white.csv", sep=";")
    meta_info = json.load(open(path + "uci_regression/wine_white/data_types.json"))
    x, y = data.iloc[:,:-1].values, data.iloc[:,[-1]].values
    return x, y, "Regression", meta_info

def load_wine_red(path="./", missing_strategy="drop"):
    
    # red wine dataset
    data = pd.read_csv(path + "uci_regression/wine_red/wine_red.csv", sep=";")
    meta_info = json.load(open(path + "uci_regression/wine_red/data_types.json"))
    x, y = data.iloc[:,:-1].values, data.iloc[:,[-1]].values
    return x, y, "Regression", meta_info