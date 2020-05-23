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



%% Get Data if Already Exists
%TT = readtable(fullfile('..', 'data', 'RESULTS', 'hyperparameters_2.dat'));
% download data
if isfile(fullfile('..', 'data', 'RESULTS', 'hyperparameters.dat')) && ...
        isfile(fullfile('..', 'data', 'RESULTS', 'CPU.dat'))&& ...
        isfile(fullfile('..', 'data', 'RESULTS', 'NMSE.dat'))&& ...
        isfile(fullfile('..', 'data', 'RESULTS', 'Y_Prediction.dat'))
    
    
     CPU = readmatrix(fullfile('..', 'data', 'RESULTS', 'CPU.dat'));
     NMSE = readmatrix(fullfile('..', 'data', 'RESULTS', 'NMSE.dat'));
     Y_prediction = readmatrix(fullfile('..', 'data', 'RESULTS', 'Y_Prediction.dat'));
     Hyperparameters = readtable(fullfile('..', 'data', 'RESULTS', 'hyperparameters.dat'));
 
else
    %% initialize LWPR
    
    n_in = size(X,2) ;                              % : number of input dimensions
    n_out = size(Y,2) ;                             % : number of output dimensions
    
    diag_only = [1];                              % : 1/0 to update only the diagonal distance metric
    meta = [1];                                   % : 1/0 to allow the use of a meta learning parameter
    meta_rate = [0.1];                    % : the meta learning rate
    penalty = [1.e-5 1.e-3];            % : a smoothness bias, usually a pretty small number (1.e-4)
    init_alpha= [ 0.0001 0.001];                    % : the initial learning rates
    init_D= [ 1.e-4 1.e-3 1.e-2];                          % : the initial distance metrics
    w_gen = [  0.0002 ];  % : weight
    init_lambda = [ 0.0001 0.001 ];                           % : the initial lambda
    final_lambda = [0.99 0.999];                          % : the final lambda
    tau_lambda = [ 0.05 0.1 ];                          % : the tau lambda
    
    % best : 23.045.2020 at 4.18pm : 0.723 with
    % 18/1/1/1/0.1/1e-3/0.01/1e-3/2e-4/0.001/0.99/0.1
    % Create a matrix with all the hyperparameters
    hyperparameters = transpose(combvec(n_in, n_out, diag_only,meta,meta_rate,...
        penalty, init_alpha, init_D, w_gen, init_lambda, final_lambda, tau_lambda));
    
    
    % Set up an ID for each combinaison & add it to the matrix
    IDs = transpose([1:1:size(hyperparameters,1)]);
    hyperparameters = [IDs hyperparameters];
    
    % Transform the Matrix to a Table
    Hyperparameters = array2table(hyperparameters);
    Hyperparameters.Properties.VariableNames={'ID' 'n_in' 'n_out' 'diag_only' 'meta' ...
        'meta_rate' 'penalty' 'init_alpha' 'init_D' 'w_gen' 'init_lambda' ...
        'final_lambda' 'tau_lambda'};
   
     %% Launch LWPR Algorithm
     
     [NMSE, CPU, Y_prediction]= lwpr_test(hyperparameters,X,Y,Xt,Yt);
     
     %% Write Table
     
%      writetable(hypToTest,fullfile('..', 'data', 'RESULTS','hyperparameters.dat'),'WriteRowNames',true)
%      writematrix(NMSE,fullfile('..', 'data', 'RESULTS','NMSE.dat'))
%      writematrix(CPU,fullfile('..', 'data', 'RESULTS','CPU.dat'))
%      writematrix(Y_prediction,fullfile('..', 'data', 'RESULTS','Y_prediction.dat'))
end

%% Get the minimum nMSE and the ID
NMSE(NMSE <= 0) = NaN;
[value, index] = min(NMSE(2,:));

Hyperparameters(index,:).ID
fprintf('#ID = %d nMSE=%5.3f',Hyperparameters(index,:).ID,value);

Prediction_Plot=table(data.Date, data.y_t, cat(1,Y,Y_prediction(:,index)));
Prediction_Plot.Properties.VariableNames={'Date' 'Y' 'Y_Prediction'};
%% Plot
fig= figure();
plot(Prediction_Plot.Date,Prediction_Plot.Y_Prediction, 'r','LineWidth',1.2)
hold on
plot(Prediction_Plot.Date,Prediction_Plot.Y,'b','LineWidth',1.2)
set(gca,'FontSize',16)
title('Actual Portfolio Return vs Predicted Portfolio Return using LWPR','FontSize', 20)
xlabel('Date','FontSize', 16)
ylabel('Daily Return','FontSize', 16)
xline(Prediction_Plot.Date(n),'LineWidth',3)
legend({'Prediction' 'Actual Return' 'Start of Test Set'})
hold off
saveas(fig,fullfile('..', 'figures','actualvspredicted.png'));

% %%
% IDs= Hyperparameters.ID;
% T_CPU = array2table([IDs transpose(CPU)]);
% T_CPU.Properties.VariableNames={'ID' 'TrainCPU' 'TestCPU'};
% T_nMSE = array2table([IDs transpose(NMSE)]);
% T_nMSE.Properties.VariableNames={'ID' 'TrainnMSE' 'TestnMSE'};
% TableModelsAnalyse=join(Hyperparameters,join(T_CPU,T_nMSE));
% %%
% t_ = Hyperparameters.Properties.VariableNames(4:end);
% fig2=figure();
% for p =1:length(t_)
%     subplot(5,2,p);
%     boxplot([TableModelsAnalyse{:,14}],TableModelsAnalyse{:,t_{p}})
%     title(sprintf('Parameter: %s', t_{p}))
%     ylabel('CPU cons. (s)')
% end
% sgtitle('Train CPU / hyperparameter', 'FontSize', 20);
% %%
% fig3=figure();
% for p =1:length(t_)
%     subplot(5,2,p);
%     boxplot([TableModelsAnalyse{:,17}],TableModelsAnalyse{:,t_{p}})
%     title(sprintf('Parameter: %s', t_{p}))
%     ylabel('Test nMSE')
%     ylim([0.72 0.8])
% end
% sgtitle('Test nMSE / hyperparameter', 'FontSize', 20);