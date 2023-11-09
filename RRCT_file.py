# Function to obtain feature ranking (and select features) using RRCT


# from RRCT_file import RRCT

# Indicative approach to load a dataset and run RRCT
"""
from scipy.io import loadmat
A = loadmat('D:/DATA for publications/DATASETS/synthetic_dataset3_Patterns_Tsanas.mat')
X = A['X']; y = A['y']
features_GT = A['features_GT'] -1 #RR!! use -1: [MATLAB starts counting at 1; Python starts at 0]
K=len(X[0])
from RRCT_file import RRCT # import function to call later
features, RRCT_all = RRCT(X, y, K=30); # Run the RRCT algorithm to obtain the feature ranking for the top-30 features
"""


# Proceed with the definition of the function
def RRCT(X, y, K=0):
    """
    General call: 
        features = RRCT(X, y);
        features, RRCT_all = RRCT(X, y);
        features, RRCT_all = RRCT(X, y, 30);    

    % Function for feature selection (FS), based on the RRCT concept
    RELEVANCE, REDUNDANCY AND COMPLEMENTARITY TRADE-OFF (RRCT)

    This function is based on solid theoretical principles, integrating the
    key concepts for FS about feature relevance, redundancy, and conditional
    relevance (complementarity)

    Aim: select the best features out of a matrix NxM

    Inputs:  X       -> N by M matrix, N = observations, M = features
             y       -> N by 1 vector with the numerical outputs
    __________________________________________________________________________
    optional inputs:
             K       -> number of features to be selected (integer>0)               [default = M, the dimensionality of the dataset]

    =========================================================================
    Output:
           features  -> Selected feature subset in descending order
           RRCT_all  -> All the outputs in a struct:relevance, redundancy,
                         complementarity and FS algorithm's output
    =========================================================================

    Part of the "Tsanas FSToolbox"

    -----------------------------------------------------------------------
    Useful references:

    1)  A. Tsanas: "Relevance, redundancy and complementarity trade-off (RRCT):
        a generic, efficient, robust feature selection tool", Patterns,
        (in press), 2022
    2)  A. Tsanas: "Accurate telemonitoring of Parkinson's disease symptom
        severity using nonlinear speech signal processing and statistical
        machine learning, D.Phil. thesis, University of Oxford, UK, 2012
    3)  R. Battiti: Using mutual information for selecting features in
        supervised neural net learning, IEEE Transactions on Neural Networks,
        Vol. 5(4), pp. 537–550, 1994
    4)  H. Peng, F. Long, and C. Ding: Feature selection based on mutual
        information: criteria of max-dependency, max-relevance, and
        min-redundancy,IEEE Transactions on Pattern Analysis and Machine
        Intelligence, Vol. 27, No. 8, pp.1226-1238, 2005
    -----------------------------------------------------------------------

    Modification history
    --------------------
    1 February 2022: Porting MATLAB code to Python, development of the function

    -----------------------------------------------------------------------
    (c) Athanasios Tsanas, 2022

    ********************************************************************
    If you use this program please cite:

    1)  A. Tsanas: "Relevance, redundancy and complementarity trade-off (RRCT):
        a generic, efficient, robust feature selection tool", Patterns,
        (in press), 2022
    ********************************************************************

    For any question, to report bugs, or just to say this was useful, email
    tsanasthanasis@gmail.com

    *** For updates please check: https://github.com/ThanasisTsanas/RRCT ***

    ========================== *** LICENSE *** ==============================

    In short, you are free to use the software for academic research; for commercial pursposes please contact me:
                        A. Tsanas: tsanasthanasis@gmail.com

    ========================== *** LICENSE *** ==============================
    """

    # %% ======================================================================
    #                     Load libraries and set defaults

    # Load libraries in Python (the following are quite standard)
    import numpy as np  # import Numpy (numerical python library)
    # import matplotlib.pyplot as plt # import a standard interface to provide MATLAB-like plotting
    import pandas as pd  # import pandas, well used in ML
    import scipy.stats
    import pingouin as pg  # use the partial_corr function for the computation of the partial correlation
    import time
    # from collections import namedtuple

    N, M = np.shape(X)

    # guard against the case the user has indicated more features than the dimensionality of the dataset (or less than 1)
    if (K < 1):
        K = M
    if (K > M):
        K = M
        print(
            'You provided K>M, i.e. more features to be ranked than the dimensionality of the data, reverting to K=M!')

        # Get Code Block Start Time
    start_time = time.perf_counter()

    # %% =======================================================================
    #                           Data preprocessing

    X_standardized = scipy.stats.zscore(X)

    X_df = pd.DataFrame(
        X_standardized)  # Converting array to pandas DataFrame; makes processing simpler working with pandas
    y_df = pd.DataFrame(y)

    data_Xy = X_df;
    data_Xy = data_Xy.assign(y=y_df)  # Dataframe bringing together X and y

    # %% ======================================================================
    #            Compute relevance, redundancy and build the mRMR matrix

    relv = X_df.corrwith(y_df.squeeze(),
                         method="spearman")  # correlation coefficient for each of the variables in X with y: squeeze dataframe into Series
    relevance = -0.5 * np.log(1 - relv ** 2)  # **** RELEVANCE BY DEFINITION

    redundancy = X_df.corr(method="spearman")  # compute correlation matrix using Spearman's method
    redundancy = redundancy - np.identity(M)
    redundancy = -0.5 * np.log(1 - redundancy ** 2)  # **** REDUNDANCY BY DEFINITION

    # if we want to visualize the resulting redundancies
    # sb.heatmap(redundancy, annot=True) # heatmap to visualize correlation matrix
    # plt.title('Heatmap of feature redundancies', fontsize=14, fontweight='bold')
    # plt.show()

    # Define convenient matrix
    mRMR_matrix = redundancy
    mRMR_matrix = mRMR_matrix + np.diag(relevance)

    # initialize vectors that will hold the relevance, redundancy and complementarity values
    RRCT_metric = np.empty((1, K));
    RRCT_metric.fill(np.nan)
    RRCT_all_relevance = np.empty((1, K));
    RRCT_all_relevance.fill(np.nan)
    RRCT_all_redundancy = np.empty((1, K));
    RRCT_all_redundancy.fill(np.nan)
    RRCT_all_complementarity = np.empty((1, K));
    RRCT_all_complementarity.fill(np.nan)
    # features = np.empty((1,K), dtype="uint16")#; features.fill(np.nan)
    features = [''] * K

    RRCT_metric[0, 0] = relevance.max(skipna=True)

    RRCT_all_relevance[0, 0] = RRCT_metric[0, 0]
    RRCT_all_redundancy[0, 0] = 0
    RRCT_all_complementarity[0, 0] = 0

    features[0] = relevance.idxmax(skipna=True)

    Z = list();
    Z.append(features[0])

    # %% =======================================================================
    # Main loop to obtain the feature subset

    for k in range(1, K):
        # print('Working on selecting feature', k)

        candidates = pd.Series(np.arange(M))
        candidates = candidates.drop(Z)  # potential candidates at this step

        mean_redundancy = np.mean(mRMR_matrix.iloc[candidates, Z],
                                  axis=1)  # compute average redundancy for the kth step

        # complementarity = pg.partial_corr(data = data_Xy, x = candidates.values, y='y', covar = features[0, np.arange(k)], method='spearman');
        # complementarity = pg.partial_corr(data = data_Xy, x = np.array_str(v), y='y', covar = '4', method='spearman');

        df_agg = list()

        for i in candidates.index:
            comple_pair = pg.partial_corr(data=data_Xy, x=i, y='y', covar=Z,
                                          method='spearman')  # the controlling variables imm0rtal99 must be in list format
            df_agg.append(comple_pair)

        df_ngram = pd.concat(df_agg, ignore_index=True)
        complementarity = df_ngram.r  # recover all pair partial correlation computations, in Series format

        Csign = np.sign(complementarity.values) * np.sign(complementarity.values - relv[candidates])
        complementarity = -0.5 * np.log(1 - (complementarity) ** 2)
        complementarity.index = Csign.index

        # RRCT HEART: Max relevance - min redundancy + complementarity optimization
        RRCT_heart = relevance[candidates] - mean_redundancy + Csign * complementarity.values
        RRCT_metric[0, k] = RRCT_heart.max();
        fs_idx = RRCT_heart.idxmax()

        features[k] = fs_idx
        Z.append(features[
                     k])  # used to condition upon when computing the partial correlations (in subsequent for-loop steps)

        # store the three elements: Relevance, Redundancy, Complementarity
        RRCT_all_relevance[0, k] = relevance[features[k]]
        RRCT_all_redundancy[0, k] = mean_redundancy[fs_idx]
        #### RRCT_all_complementarity[0,k] = Csign[fs_idx]*complementarity[fs_idx];   ##### RR!! NEED TO FIX THIS!!!     
        RRCT_all_complementarity[0, k] = Csign.get(key=fs_idx) * complementarity.get(key=fs_idx)

    # %% =======================================================================
    # recover outputs of the function
    RRCT_all = dict([('relevance', RRCT_all_relevance), ('redundancy', RRCT_all_redundancy),
                     ('complementarity', RRCT_all_complementarity), ('RRCT_metric', RRCT_metric)])
    # features = features + 1 # this is to overcome the complication that Python starts counting at 0; for compatibility with MATLAB outputs

    # Get Code Block End Time
    end_time = time.perf_counter()
    print(f"Start Time : {start_time}")
    print(f"End Time : {end_time}")
    print(f"Total Execution Time : {end_time - start_time:0.4f} seconds\n")

    # %% =======================================================================
    # function output
    # RRCT_output = namedtuple("RRCT_output", ["ranked_features", "RRCT_all"]); return RRCT_output(features, RRCT_all)
    return features, RRCT_all