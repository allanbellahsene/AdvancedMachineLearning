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
meta_rate = [0.1 1 10 100 1000];                    % : the meta learning rate
penalty = [1.e-4 1.e-5 1.e-6 1.e-7];            % : a smoothness bias, usually a pretty small number (1.e-4)
init_alpha= [0.1 1  100 1000];                    % : the initial learning rates
init_D= [0.1 1 25 50 200];                          % : the initial distance metrics
w_gen = [0.0001 0.00015 0.0002 0.00025 0.0003];  % : weight
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
%rows = (T.diag_only==1 & T.meta==1 & T.meta_rate==100 & T.init_alpha==100 & T.init_D==25);
hypToTest = T(:,:);
hyperparameters=table2array(hypToTest);

%% Launch LWPR Algorithm

[NMSE, CPU, Y_prediction]= lwpr_test(hyperparameters,X,Y,Xt,Yt);

%% Write Table

writetable(hypToTest,fullfile('..', 'data', 'RESULTS','hyperparameters.dat'),'WriteRowNames',true)  
writematrix(NMSE,fullfile('..', 'data', 'RESULTS','NMSE.dat')) 
writematrix(CPU,fullfile('..', 'data', 'RESULTS','CPU.dat'))  
writematrix(Y_prediction,fullfile('..', 'data', 'RESULTS','Y_prediction.dat'))  
%% Get the minimum nMSE and the ID

[value, index] = min(NMSE(2,:));
[row, col] = ind2sub(size(NMSE), index);
hypToTest(col,:).ID
fprintf('#ID = %d nMSE=%5.3f',hypToTest(col,:).ID);

T_Plot=table(data.Date, data.y_t, cat(1,Y,Y_prediction(:,col)));
T_Plot.Properties.VariableNames={'Date' 'Y' 'Y_Prediction'};

%% Plot
fig= figure();
plot(T_Plot.Date,T_Plot.Y_Prediction, 'r','LineWidth',1.2)
hold on
plot(T_Plot.Date,T_Plot.Y,'b','LineWidth',1.2)
set(gca,'FontSize',16)
title('Actual Portfolio Return vs Predictied Portfolio Return using LWPR','FontSize', 20)
xlabel('Date','FontSize', 16)
ylabel('Daily Return','FontSize', 16)
xline(T_Plot.Date(n),'LineWidth',3)
legend({'Prediction' 'Actual Return' 'Start of Test Set'})
hold off
saveas(fig,fullfile('..', 'figures','actualvspredicted.png'));