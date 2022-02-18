# -*- coding: utf-8 -*-
"""
Demo file to demonstrate the use of RRCT

This demo file demonstrates two options (presented as cells below):
    (1) loads a dataset that I had originally stored in *.mat format (MATLAB's
        native format), determines the variables that contain the design matrix
        (X) and the response (y), and performs feature selection using RRCT
    (2) loads a dataset that was stored in Excel format, places the data in the 
        the standard design matrix (X) and response (y) format and performs 
        feature selection using RRCT
    
You will need to download the external 'pingouin' package before running RRCT
(this is used for the computations of the partial correlation coefficients)
https://pypi.org/project/pingouin/ 

(c) A. Tsanas, 2022
"""

# %% =========================================================================
# If the data is provided in *.mat format and want to run RRCT

from scipy.io import loadmat # required library to load files stored in *.mat format (MATLAB's native format)
A = loadmat('synthetic_dataset2_Patterns_Tsanas.mat')
X = A['X']; y = A['y'] # my suggestion is always to store data in X and y form
features_GT = A['features_GT'] -1 #RR!! use -1: [MATLAB starts counting at 1; Python starts at 0]
K=len(X[0]) # this sets the number of features to be selected to be equal to the dimensionality of the data

from RRCT_file import RRCT # import function in Python's path so that we can call it. Remember to place the RRCT file in the Python path

features, RRCT_all = RRCT(X, y, K=30); # Run the RRCT algorithm to obtain the feature ranking for the top-30 features
# The variable 'features', which is the output of RRCT, contains in descending order the selected features


# %% =========================================================================
# Alternatively, if the data was in Excel format and want to run RRCT

# Remember need to go to Python's path where the file is held
import pandas as pd # import pandas, well used in ML

# depending on the way the data is formatted in the excel file you may need to adjust the following. 
# In my examples where I used separate excel spreadsheets (X and y), this becomes:
X = pd.read_excel('synthetic_dataset2_Patterns_Tsanas.xlsx', 'X')
y = pd.read_excel('synthetic_dataset2_Patterns_Tsanas.xlsx', 'y')

# import function in Python's path so that we can call it. Remember to place the RRCT file in the Python path
from RRCT_file import RRCT

# now call RRCT
features, RRCT_all = RRCT(X, y, K=50);