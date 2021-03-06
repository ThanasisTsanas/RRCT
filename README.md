# RRCT
Relevance, Redundancy, and Complementarity Trade-off, a robust feature selection algorithm (MATLAB and Python versions)

****************************************
This function is a computationally efficient, robust approach for feature selection using the RRCT algorithm. The algorithm can be thought of as a natural extension to the popular mRMR feature selection algorithm, given that RRCT excplicitly takes into account relevance and redundancy (like mRMR), and also introduces an additional third term to account for conditional relevance (also known as complementarity).

The RRCT algorithm is computationally very efficient and can run within a few seconds including on massive datasets with thousands of features. Moreover, it can serve as a useful 'off-the-shelf' feature selection algorithm because it generalizes well on both regression and classification problems, also without needing further adjusting for mixed-type variables.

More details can be found in my paper: 

A. Tsanas: "Relevance, redundancy and complementarity trade-off (RRCT): a generic, efficient, robust feature selection tool", _Patterns_, Vol. 3:100471, 2022
https://www.cell.com/patterns/pdf/S2666-3899(22)00051-4.pdf
****************************************

**% General call: features = RRCT(X, y)**

%% Function to select the optimal features using the mRMR_Spearman approach

Inputs:  
X        -> N by M matrix, N = observations, M = features

y        -> N by 1 vector with the numerical/categorical outputs

optional inputs:  

K        -> number of features to be selected (integer value)    [default K = M, or K = 30 if there are more than 30 features in the dataset]


Output:  
 
features -> The selected features in descending order of importance (ranked by column index)

****************************************

Copyright (c) Athanasios Tsanas, 2022

**If you use this program please cite:**

A. Tsanas: "Relevance, redundancy and complementarity trade-off (RRCT): a generic, efficient, robust feature selection tool", _Patterns_, Vol. 3:100471, 2022
https://doi.org/10.1016/j.patter.2022.100471
