% Demo file to demonstrate the use of RRCT

% first load a dataset to work on (conveniently comes directly in X,y format)
load('synthetic_dataset2_Patterns_Tsanas.mat')

% now call the RRCT function
[features, RRCT_all] = RRCT(X, y, 30);

%% Load data and run RRCT if data provided in excel spreadsheet

% first load the data
X = xlsread('synthetic_dataset2_Patterns_Tsanas.xlsx', 'X');
y = xlsread('synthetic_dataset2_Patterns_Tsanas.xlsx', 'y');

% recently Matlab does not recommend the use of 'xlsread' and they indicate
% the use of 'readmatrix'. The main point is to bring the dataset in the
% format of a design matrix (which I call X), and a response variable
% vector (which I call y) -- then run directly RRCT

% now call the RRCT function
[features, RRCT_all] = RRCT(X, y, 30);