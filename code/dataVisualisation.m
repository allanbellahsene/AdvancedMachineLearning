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
% Create Matrix with t-1
X_T = innerjoin(portfolio_Return, yields,'LeftKeys',1,'RightKeys',1);
X_T = innerjoin(X_T, indices_Returns,'LeftKeys',1,'RightKeys',1);
X_T = innerjoin(X_T, commodities_Returns,'LeftKeys',1,'RightKeys',1);
X_T = innerjoin(X_T, xrate_Changes,'LeftKeys',1,'RightKeys',1);
T=X_T(:, contains(X_T.Properties.VariableNames, {'Date' 'y_t'}));
T(1,:)=[];
columns=X_T.Properties.VariableNames;
X_T.y_t=[];

X_m=table2array(X_T(1:end-1,2:end));
dataTable=[T, array2table(X_m)];
dataTable.Properties.VariableNames=columns;

writetable(dataTable,fullfile('..', 'data', 'CLEANED','cleaned_data.dat'),'WriteRowNames',true)  
%% Data visualisation

figure()
set(gcf, 'Position',  [500, 500, 1000, 1000])
for i=1:21
    subplot(7,3,i)
    plot(table2array(dataTable(:,1)), table2array(dataTable(:,i+1))) 
    title(sprintf('%s', columns{i+1}))
end
