"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import entropy, information_gain, gini_index, regression_impurity

np.random.seed(42)

# class for the tree node 
class Node:
    def __init__(self, decision_attr = None, value = None, depth = None):
        self.value = value              # leaf value of node   
        self.depth = depth              # depth of current node in tree
        self.decision_attr = decision_attr  # attribute to split on 
        self.child_nodes = {}           # dict of child nodes 
        self.prob = None                # for classification 
        self.mean = None                # for regression  

    def display_node(self, depth_indent = 0):
        """
        Recursive function to print the tree 
        """

        # lookup table for printing (for regression)
        lookup = {
            "low": "<", 
            "high": ">"
            }      

        # if not a leaf node
        if (self.decision_attr != None):
            for cn in self.child_nodes:
                # for classification
                if (self.child_nodes[cn].prob != None): 
                    print("|   " * depth_indent + "| ?(X({}) = {}):".format(self.decision_attr, cn))
                
                # for regression 
                else:
                    print("|   " * depth_indent + "| ?(X({}) {} {:.2f}):".format(self.decision_attr, lookup[cn], self.mean))
                
                self.child_nodes[cn].display_node(depth_indent + 1)
        
        # if a leaf node 
        else:
             # if value attr is str
            if (isinstance(self.value, str)):
                print("|   " * depth_indent + "|--- Value = {} Depth = {}".format(self.value, self.depth))

            # int or float (as we need to round-off here)
            else: 
                print("|   " * depth_indent + "|--- Value = {:.2f} Depth = {}".format(self.value, self.depth))
        
    def getVal(self, X, max_depth=np.inf):
        '''
        Recursive function to check the input data and return the value stored 
        at leaf node

        'max-depth' : maximum depth to raverse in the tree
        default value : np.inf => traverse the entire tree 
        '''

        # if a leaf node or max depth reached 
        if (self.decision_attr == None or self.depth >= max_depth):
            return self.value 

        # if not a leaf node
        else:
            # for classification, mean == None
            if (self.mean == None):
                # input feature already seen in training data
                if (X[self.decision_attr] in self.child_nodes):
                    next_level = self.child_nodes[X[self.decision_attr]]
                    # drop the attr used to split 
                    return next_level.getVal(X.drop(self.decision_attr), max_depth=max_depth)
                 
                else:
                    max_prob_child, max_prob = max(self.child_nodes.items(), key=lambda x:x[1].prob)
                    return self.child_nodes[max_prob_child].getVal(X.drop(self.decision_attr), max_depth=max_depth)


            # for regression
            else:
                # comparing value to the mean of current node and 
                # returning valur of appropriate child node  
                if (X[self.decision_attr] <= self.mean):
                    return self.child_nodes["low"].getVal(X, max_depth=max_depth)
                else:
                    return self.child_nodes["high"].getVal(X, max_depth=max_depth)


@dataclass
class DecisionTree:
    # criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    # max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion="information_gain", max_depth=10):
        self.root = None                    # root node 
        self.max_depth = max_depth          # max depth tree can grow to, defaul value = 10
        self.task_type = None               # to determine classification or regression 
        
        self.criterion = criterion          # determines the best split 
        self.n_samples = None               # len(X) => number of rows in X, track the no of sampels
        self.colname = None                 # store column names in X 

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        
        # Storing dataset parameters
        self.n_samples = len(X)
        self.task_type = y.dtype
        self.colname = y.name

        # learning tree
        # stored in root attribute
        self.root = self.build_tree(X, y, None)
        self.root.prob = 1
        pass

    def build_tree(self, X, Y, parent_node, depth = 0):
        '''
        Recursive function to build tree.
        X: features
        y: labels
        parent_node: caller of the function 
        depth: current depth
        '''
        
        # edge case handling: 
        # only one class left in labels (Y)
        if (Y.nunique() == 1):
            return Node(value = Y.values[0], depth = depth)
            # return Node(value = Y.iloc[0], depth = depth)

        # if feature dataset empty or max depth reached or all features have same values 
        if (len(X.columns) <= 0 or depth >= self.max_depth or len(list(X.columns)) == sum(list(X.nunique()))):
            # classification
            if str(Y.dtype) == 'category': 
                return Node(value=Y.mode(dropna=True)[0], depth = depth)
            # regression 
            else: 
                return Node(value=Y.mean(), depth = depth)


        # calculating information gain
        max_ig = - np.inf # highest IG
        max_mean = None # mean of the feature with highest IG

        # determining the best column 
        for c in list(X.columns): 
            # for classification
            if (str(Y.dtype) == "category"):
                if (self.criterion == "information_gain"):
                    column_ig = information_gain(Y, X[c])
                elif (self.criterion == "gini_index"):
                    column_ig = gini_gain(Y, X[c])

            # for regression
            else:
                column_ig = regression_impurity(Y, X[c]) 

            # if attribute selected is range of values
            if (type(column_ig) == tuple):
                mean_val = column_ig[1]
                column_ig = column_ig[0]

            if (column_ig > max_ig):
                max_ig = column_ig 
                best_split_col = c # name of the column with highest IG (best split)
                max_mean = mean_val 


        # creating a new node based on best column
        node = Node(best_split_col=best_split_col)
        # best column data
        best_col_data = X[best_split_col]       
        
        # for discrete
        if (str(best_col_data.dtype) == "category"):
            X = X.drop(best_split_col, axis=1) # to avoid overfitting

            # group unique values 
            best_split_col_classes = best_col_data.groupby(best_col_data).count() 

            for catVal, count in best_split_col_classes.items():
                frows = (best_col_data == catVal) # bool mask to filter rows
                if (count > 0):
                    node.child_nodes[catVal] = self.build_tree(X[frows], Y[frows], node, depth+1) 
                    node.child_nodes[catVal].prob = len(X[frows])/self.X_len 

        # for continuous/real
        else:
            # filtering rows based on max_mean
            low_ind = (best_col_data <= max_mean) 
            high_ind = (best_col_data >= max_mean)

            # creating child nodes on the current node
            node.child_nodes["low"] = self.build_tree(X[low_ind], Y[low_ind], node, depth+1)
            node.child_nodes["high"] = self.build_tree(X[high_ind], Y[high_ind], node, depth+1)
            
             # mean of the best_split_col
            node.mean = max_mean  

        # node values
        if str(Y.dtype) == 'category': node.value = Y.mode(dropna=True)[0] 
        else: node.value = Y.mean()

        # node depth
        node.depth = depth

        return node


    def predict(self, X: pd.DataFrame, max_depth = np.inf) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        y_pred = [] # predicted values for each row in X 

        for i in X.index: 
            # make prediction for each row in X 
            # append to answer
            y_pred.append(self.root.getVal(X.loc[i], max_depth = max_depth)) 
        
        # return a series of predicted values 
        return pd.Series(y_pred, name = self.colname).astype(self.task_type)

        pass

    
    def plot(self, node=None, depth=0) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        self.root.display_node() 
