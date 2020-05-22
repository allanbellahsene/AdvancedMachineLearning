%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=======================================================================================================%
%====================================== Advanced Machine Learning ======================================%
%========================================= Team O - SVR vs LWPR ========================================%
%==================================== BRODARD Lionel, BELLAHSENE Allan =================================%
%================================================ main =================================================%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;

addpath(genpath('functions'))                   % add function apth
rng default                                     % keep same random numbers

%% Imports data
data = readtable(fullfile('..', 'data', 'CLEANED', 'cleaned_data.dat'));

%% Train & Test Data

n= round(height(data)*0.7);

% Train Data
X = table2array(data(1:n,3:end));
Y = data.y_t(1:n);

% Test Data
Xt = table2array(data(n+1:end,3:end));
Yt = data.y_t(n+1:end);

%% initialize LWPR

n_in = size(X,2) ;                              % : number of input dimensions
n_out = size(Y,2) ;                             % : number of output dimensions

diag_only = [0 1];                              % : 1/0 to update only the diagonal distance metric
meta = [0 1];                                   % : 1/0 to allow the use of a meta learning parameter
meta_rate = [1 10 100 1000];                    % : the meta learning rate
penalty = [1.e-4 1.e-5 1.e-6 1.e-7];            % : a smoothness bias, usually a pretty small number (1.e-4)
init_alpha= [1 10 100 1000];                    % : the initial learning rates
init_D= [1 25 50 200];                          % : the initial distance metrics
w_gen = [0.0001 0.00025 0.0005 0.00075 0.001];  % : weight
init_lambda = [0.99];                           % : the initial lambda
final_lambda = [0.99];                          % : the final lambda
tau_lambda = [0.99];                            % : the tau lambda


% Create a matrix with all the hyperparameters
hyperparameters = transpose(combvec(n_in, n_out, diag_only,meta,meta_rate,...
    penalty, init_alpha, init_D, w_gen, init_lambda, final_lambda, tau_lambda));

% Set up an ID for each combinaison & add it to the matrix
IDs = transpose([1:1:length(hyperparameters)]);
hyperparameters = [IDs hyperparameters];

% Transform the Matrix to a Table
T = array2table(hyperparameters);
T.Properties.VariableNames={'ID' 'n_in' 'n_out' 'diag_only' 'meta' ...
    'meta_rate' 'penalty' 'init_alpha' 'init_D' 'w_gen' 'init_lambda' ...
    'final_lambda' 'tau_lambda'};


%% Hyperparameters to Test

% Choose specific hyperparameters to test
rows = (T.diag_only==1 & T.meta==1 & T.meta_rate==100 & T.init_alpha==100 & T.init_D==25);
hypToTest = T(rows,:);
hyperparameters=table2array(hypToTest);

%% Launch LWPR Algorithm

[NMSE, CPU, Y_prediction]= lwpr_test(table2array(hypToTest),X,Y,Xt,Yt);

%% Get the minimum nMSE and the ID

[value, index] = min(NMSE(2,:));
[row, col] = ind2sub(size(NMSE), index);
hypToTest(col,:).ID
fprintf('#ID = %d nMSE=%5.3f',hypToTest(col,:).ID);

%% Plot

plot(Y_prediction(:,col))
hold on
plot(Yt)
hold off
%%
plot(data.Date,data.y_t)