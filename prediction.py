import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import preprocessing


def final_fun_1(X):
    """
    This function includes the entire pipeline, from data preprocessing to making final predictions.
    Input : Raw Data
    Output: Predictions for the input
    """ 
    # Preprocessing
    preprocessed = preprocessing.preprocessing(X)
    # Standardize train features by removing the mean and scaling to unit variance.
    scaler = StandardScaler()
    X = scaler.fit_transform(preprocessed.T)
    X = pd.DataFrame(X.T)
    # Load the best model
    filename = "bestmodel.sav"
    modellist = pickle.load(open(filename, 'rb'))
    clf = modellist[1]
    pred = clf.predict(X) 
    return pred