from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    # TODO: Write here
    pass


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    # Following asserts added
    assert y_hat.size == y.size         # check if sizes of y_hat and y are equal
    assert y_hat.name == y.name         # names of input series should be same
    assert y_hat.dtype == y.dtype       # data types of input series should be same
    assert isinstance(cls, (int, str))  # cls should be either int or str

    tp = ((y_hat == cls) & (y == cls)).sum()    # true positives 
    pp = (y_hat == cls).sum()                   # predicted positives (true positives + false positives)

    return tp / pp      # precision = tp / (tp + fn)

    pass


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """

    assert y_hat.size == y.size 
    assert y_hat.name == y.name 
    assert y_hat.dtype == y.dtype 
    assert isinstance(cls, (int, str))

    tp = ((y_hat == cls) & (y == cls)).sum() # true positives
    fn = ((y_hat != cls) & (y == cls)).sum() # false negatives

    return tp / (tp + fn) # recall = tp / (tp + fn) 

    pass


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """

    assert y_hat.size == y.size 
    assert y_hat.name == y.name 
    assert y_hat.dtype == y.dtype 

    return ((y_hat - y) ** 2).mean() ** 0.5     # rmse = sqrt(mean((y_hat - y) ** 2))

    pass


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """

    assert y_hat.size == y.size 
    assert y_hat.name == y.name 
    assert y_hat.dtype == y.dtype

    return (y_hat - y).abs().mean()     # mae = mean(abs(y_hat - y))

    pass
