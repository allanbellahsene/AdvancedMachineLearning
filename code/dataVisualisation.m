%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=======================================================================================================%
%====================================== Advanced Machine Learning ======================================% 
%========================================= Team O - SVR vs LWPR ========================================%
%==================================== BRODARD Lionel, BELLAHSENE Allan =================================%
%========================================== Data - Visualisation =======================================%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;

%% Imports data

funds_Returns = readtable(fullfile('..', 'data', 'CLEANED', 'Funds_Returns.dat'));
commodities_Returns = readtable(fullfile('..', 'data', 'CLEANED', 'commodities_Return.dat'));
yields = readtable(fullfile('..', 'data', 'CLEANED', 'Yields.dat'));
indices_Returns = readtable(fullfile('..', 'data', 'CLEANED', 'marketIndices_Returns.dat'));
xrate_Changes = readtable(fullfile('..', 'data', 'CLEANED', 'exchangeRates_Changes.dat'));

%% Create Equal Weighted Portfolio

portfolio_Return(:,1)=funds_Returns(:,1); 
portfolio_Return(:,2)=array2table(1/5*(table2array(funds_Returns(:,2))+table2array(funds_Returns(:,3)) +table2array(funds_Returns(:,4)) +table2array(funds_Returns(:,5)) +table2array(funds_Returns(:,6)) ));
portfolio_Return.Properties.VariableNames={'Date' 'y_t' };

%%
% Join Yields
X_Matrix = innerjoin(portfolio_Return, yields,'LeftKeys',1,'RightKeys',1);
X_Matrix = innerjoin(X_Matrix, indices_Returns,'LeftKeys',1,'RightKeys',1);

