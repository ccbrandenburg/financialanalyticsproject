# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 11:39:04 2016
@author: rahulmehra
"""



# Import the modules
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Define a function to autoclean the pandas dataframe
def autoclean(x):

    for column in x.columns:

        # Replace NaNs with the median or mode of the column depending on the column type
        try:
            x[column].fillna(x[column].median(), inplace=True)
        except TypeError:
            x[column].fillna(x[column].mode(), inplace=True)

        # Select the columns with type int and float
        if x[column].dtypes == 'int64' or x[column].dtypes == 'float64':

            #Calculate mean of the column
            mean = x[column].mean()

            #Calculate the standard deviation of the column
            std = 2.5*x[column].std()

            # See for the outliers and impute with median
            x[column] = x[column].apply(lambda y: x[column].median() if(abs(y - mean >std)) else y)

            # Calculate the number of rows in dataframe
            n_rows = len(x.index)

            #Calculate the percentage of negative values in the column
            negative_perc = np.sum((x[column] < 0))/n_rows

            #Handle the unreliable values (like negative values in the positive value column)
            x[column] = x[column].apply(lambda y: -(y) if (y<0 and negative_perc >= 0.05) else y)

        # Encode all strings with numerical equivalents
        if str(x[column].values.dtype) == 'object':
            column_encoder = LabelEncoder().fit(x[column].values)

            x[column] = column_encoder.transform(x[column].values)
    


    return(x)
