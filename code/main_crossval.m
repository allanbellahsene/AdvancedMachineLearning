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
  
    diag_only = [0 1];                              % : 1/0 to update only the diagonal distance metric
    meta = [0 1];                                   % : 1/0 to allow the use of a meta learning parameter
    meta_rate = [ 0.001 0.01 0.1 0.15];                    % : the meta learning rate
    penalty = [ 1.e-5 1.e-4 1.e-3];            % : a smoothness bias, usually a pretty small number (1.e-4)
    init_alpha= [ 1.e-5 1.e-4 1.e-3 1.e-2 ];                    % : the initial learning rates
    init_D= [ 1.e-4 1.e-3 1.e-2 1.e-1];                          % : the initial distance metrics
    w_gen = [ 1.e-4 1.5e-4 2.e-4];  % : weight
    init_lambda = [1.e-5 1.e-4 1.e-3 ];                           % : the initial lambda
    final_lambda = [0.99 0.999 0.9999];                          % : the final lambda
    tau_lambda = [0.01 0.05 0.1];                            % : the tau lambda
    %% Initial Train & Test
    
    % Initial n
    n= round(height(data)*0.7);
    
    % Initial Train Data
    X = table2array(data(1:n,3:end));
    Y = data.y_t(1:n);
    
    % Initial Test
    Xt = table2array(data(n+1:n+1,3:end));
    Yt = data.y_t(n+1:n+1);
    
    n_in = size(X,2) ;                              % : number of input dimensions
    n_out = size(Y,2) ;                             % : number of output dimensions
    
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
      % Initial n
    n= round(height(data)*0.7);
    nn = height(data)-n;
    NMSE_3D=zeros(2,length((hyperparameters(:,1))),nn);
    CPU_3D=zeros(2,length((hyperparameters(:,1))),nn);
    Y_prediction = zeros(nn,length((hyperparameters(:,1))));
    for j = 1:nn
        % Train Data
        X = table2array(data(1:n+j-1,3:end));
        Y = data.y_t(1:n+j-1);
        % Initial Test
        Xt = table2array(data(n+j:n+j,3:end));
        Yt = data.y_t(n+j:n+j);
        [NMSE_3D(:,:,j), CPU_3D(:,:,j), Y_prediction(j,:)]= lwpr_test_CV(hyperparameters,X,Y,Xt,Yt);
    end     
     %% Write Table
     
     writetable(Hyperparameters,fullfile('..', 'data', 'RESULTS','hyperparameters.dat'),'WriteRowNames',true)
     writematrix(NMSE_3D,fullfile('..', 'data', 'RESULTS','NMSE.dat'))
     writematrix(CPU_3D,fullfile('..', 'data', 'RESULTS','CPU.dat'))
     writematrix(Y_prediction,fullfile('..', 'data', 'RESULTS','Y_prediction.dat'))
end

%% Get the minimum nMSE and the ID
NMSE=mean(NMSE_3D,3);
CPU=mean(CPU_3D,3);

NMSE(NMSE <= 0) = NaN;
[value, index] = min(NMSE(2,:));

Hyperparameters(index,:).ID
fprintf('#ID = %d nMSE=%5.3f',Hyperparameters(index,:).ID,value);

Prediction_Plot=table(data.Date, data.y_t, cat(1,data.y_t(1:n),Y_prediction(:,index)));
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


%%
IDs= Hyperparameters.ID;
T_CPU = array2table([IDs transpose(CPU)]);
T_CPU.Properties.VariableNames={'ID' 'TrainCPU' 'TestCPU'};
T_nMSE = array2table([IDs transpose(NMSE)]);
T_nMSE.Properties.VariableNames={'ID' 'TrainnMSE' 'TestnMSE'};
TableModelsAnalyse=join(Hyperparameters,join(T_CPU,T_nMSE));
%%
t_ = Hyperparameters.Properties.VariableNames(4:end);
fig2=figure();
for p =1:length(t_)
    subplot(5,2,p);
    boxplot([TableModelsAnalyse{:,14}],TableModelsAnalyse{:,t_{p}})
    title(sprintf('Parameter: %s', t_{p}))
    ylabel('CPU cons. (s)')
end
sgtitle('Train CPU / hyperparameter', 'FontSize', 20);
%%
fig3=figure();
for p =1:length(t_)
    subplot(5,2,p);
    boxplot([TableModelsAnalyse{:,17}],TableModelsAnalyse{:,t_{p}})
    title(sprintf('Parameter: %s', t_{p}))
    ylabel('Test nMSE')
    ylim([0.72 0.8])
end
sgtitle('Test nMSE / hyperparameter', 'FontSize', 20);
%%
%
NMSE_N = zeros(nn,1);
CPU_N=zeros(nn,1);
for i =1:nn
    NMSE_N(i,1)= mean(mean(NMSE_3D(:,:,i)));
    CPU_N(i,1)= mean(mean(CPU_3D(1,:,i)));
end
train_size = transpose([n:1:n+nn-1]);

outcomeVStrainSize=[train_size NMSE_N CPU_N];

fig4=figure();
plot(outcomeVStrainSize(:,1),outcomeVStrainSize(:,2), 'r','LineWidth',1.2)
hold on
plot(outcomeVStrainSize(:,1),outcomeVStrainSize(:,3),'b','LineWidth',1.2)
set(gca,'FontSize',16)
title('Impact of N','FontSize', 20)
xlabel('Train Sample Size','FontSize', 16)
legend({'nMSE' 'CPU Time (s)'})
hold off
saveas(fig,fullfile('..', 'figures','TrainDataSizeImpact.png'));

