import pandas as pd
import numpy as np


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy 
    """
    # classes = Y.unique()
    prob = Y.value_counts()/Y.value_counts().sum()
    return (- np.sum(prob * np.log2(prob)))   # entropy = - sum(p * log2(p))

    pass


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    # classes = Y.unique()
    prob = Y.value_counts()/Y.value_counts().sum()
    return (1 - np.sum(prob**2))    # gini index = 1 - sum(p^2)

    pass


def information_gain(Y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate the information gain
    """
    # classes = Y.unique()
    prob = Y.value_counts()/Y.value_counts().sum()
    initial_entropy = -np.sum(prob * np.log2(prob)) 
    sub_entropy = 0

    values = attr.unique()   

    # calculate the entropy for each subset of the attribute 
    for val in values: 
        sub = Y[attr == val]  
        p = sub.value_counts()/sub.value_counts().sum()
        e = - np.sum(p * np.log2(p)) 

        sub_entropy += (sub.shape[0]/Y.shape[0]) * e   # weighted average of the entropy of each subset 
    
    return (initial_entropy - sub_entropy)

    pass
