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
     yPrediction = readmatrix(fullfile('..', 'data', 'RESULTS', 'Y_Prediction.dat'));
     Hyperparameters = readtable(fullfile('..', 'data', 'RESULTS', 'hyperparameters.dat'));
 
else
    %% initialize LWPR
  
    diagOnly = [0 1];                              % : 1/0 to update only the diagonal distance metric
    meta = [0 1];                                   % : 1/0 to allow the use of a meta learning parameter
    metaRate = [ 0.001 0.01 0.1 0.15];                    % : the meta learning rate
    penalty = [ 1.e-5 1.e-4 1.e-3];            % : a smoothness bias, usually a pretty small number (1.e-4)
    initAlpha= [ 1.e-5 1.e-4 1.e-3 1.e-2 ];                    % : the initial learning rates
    initD= [ 1.e-4 1.e-3 1.e-2 1.e-1];                          % : the initial distance metrics
    wGen = [ 1.e-4 1.5e-4 2.e-4];  % : weight
    initLambda = [1.e-5 1.e-4 1.e-3 ];                           % : the initial lambda
    finalLambda = [0.99 0.999 0.9999];                          % : the final lambda
    tauLambda = [0.01 0.05 0.1];                            % : the tau lambda
    
    
    %% Initial Train & Test
    
    % Initial n
    n= round(height(data)*0.7);
    
    % Initial Train Data
    X = table2array(data(1:n,3:end));
    Y = data.y_t(1:n);
    
    % Initial Test
    Xt = table2array(data(n+1:n+1,3:end));
    Yt = data.y_t(n+1:n+1);
    
    nIn = size(X,2) ;                              % : number of input dimensions
    nOut = size(Y,2) ;                             % : number of output dimensions
    
    % Create a matrix with all the hyperparameters
    hyperparameters = transpose(combvec(nIn, nOut, diagOnly,meta,metaRate,...
        penalty, initAlpha, initD, wGen, initLambda, finalLambda, tauLambda));
    
    
    % Set up an ID for each combinaison & add it to the matrix
    IDs = transpose([1:1:size(hyperparameters,1)]);
    hyperparameters = [IDs hyperparameters];
    
    % Transform the Matrix to a Table
    Hyperparameters = array2table(hyperparameters);
    Hyperparameters.Properties.VariableNames={'ID' 'nIn' 'nOut' 'diagOnly' 'meta' ...
        'metaRate' 'penalty' 'initAlpha' 'initD' 'wGen' 'initLambda' ...
        'finalLambda' 'tauLambda'};
   
     %% Launch LWPR Algorithm
      % Initial n
    n= round(height(data)*0.7);
    nn = height(data)-n;
    
    rows = ( Hyperparameters.meta==1 &...
        Hyperparameters.metaRate==0.1 & Hyperparameters.penalty==1e-3 & ...
        Hyperparameters.initAlpha==0.01 & Hyperparameters.initD==1e-3 & ...
        Hyperparameters.wGen==2e-4 & Hyperparameters.initLambda==0.001 &...
        Hyperparameters.finalLambda==0.99 & Hyperparameters.tauLambda==0.1);
    hyperparameters_table = Hyperparameters(rows,:);
    hyperparameters_array=table2array(hyperparameters_table);
    
    NMSE_3D=zeros(2,length((hyperparameters_array(:,1))),nn);
    CPU_3D=zeros(2,length((hyperparameters_array(:,1))),nn);
    yPrediction = zeros(nn,length((hyperparameters_array(:,1))));
    % Choose the Hyperparameters to run:
    % Best Run : 23.05.2020 at 4.18pm : 0.723 with
    % 18/1
    % /1/1/0.1/1e-3/0.01/1e-3/2e-4/0.001/0.99/0.1
    % Choose specific hyperparameters to test
    CV = 0;
    
%     for j = 1:nn
%         % Train Data
%         X = table2array(data(1:n+j-1,3:end));
%         Y = data.y_t(1:n+j-1);
%         % Initial Test
%         Xt = table2array(data(n+j:n+j,3:end));
%         Yt = data.y_t(n+j:n+j);
%         [NMSE_3D(:,:,j), CPU_3D(:,:,j), Y_prediction(j,:)]= lwpr_test(hyperparameters,X,Y,Xt,Yt,CV);
%     end
%     
    [NMSE_3D,CPU_3D,yPrediction] = lwpr_run(hyperparameters_array,data,CV,NMSE_3D,CPU_3D,yPrediction);
%      %% Write Table
%      
%      writetable(Hyperparameters,fullfile('..', 'data', 'RESULTS','hyperparameters.dat'),'WriteRowNames',true)
%      writematrix(NMSE_3D,fullfile('..', 'data', 'RESULTS','NMSE.dat'))
%      writematrix(CPU_3D,fullfile('..', 'data', 'RESULTS','CPU.dat'))
%      writematrix(Y_prediction,fullfile('..', 'data', 'RESULTS','Y_prediction.dat'))
end

%% Get the minimum nMSE and the ID
NMSE=mean(NMSE_3D,3);
CPU=mean(CPU_3D,3);

NMSE(NMSE <= 0) = NaN;
[value, index] = min(NMSE(2,:));

hyperparameters_table(index,:).ID
fprintf('#ID = %d nMSE=%5.3f \n',hyperparameters_table(index,:).ID,value);

Prediction_Plot=table(data.Date, data.y_t, cat(1,data.y_t(1:n),yPrediction(:,index)));
Prediction_Plot.Properties.VariableNames={'Date' 'Y' 'yPrediction'};
%% Plot
fig= figure();
plot(Prediction_Plot.Date,Prediction_Plot.yPrediction, 'r','LineWidth',1.2)
hold on
plot(Prediction_Plot.Date,Prediction_Plot.Y,'b','LineWidth',1.2)
set(gca,'FontSize',16)
set(gcf, 'Position',  [500, 500, 800, 1000])
title('Actual Portfolio Return vs Predicted Portfolio Return using LWPR','FontSize', 20)
xlabel('Date','FontSize', 16)
ylabel('Daily Return','FontSize', 16)
xline(Prediction_Plot.Date(n),'LineWidth',3)
legend({'Prediction' 'Actual Return' 'Start of Test Set'})
hold off
saveas(fig,fullfile('..', 'figures','actualvspredicted.png'));


%%
IDs= hyperparameters_table.ID;
T_CPU = array2table([IDs transpose(CPU)]);
T_CPU.Properties.VariableNames={'ID' 'TrainCPU' 'TestCPU'};
T_nMSE = array2table([IDs transpose(NMSE)]);
T_nMSE.Properties.VariableNames={'ID' 'TrainnMSE' 'TestnMSE'};
TableModelsAnalyse=join(hyperparameters_table,join(T_CPU,T_nMSE));
%%
t_ = hyperparameters_table.Properties.VariableNames(4:end);
fig2=figure();
set(gcf, 'Position',  [500, 500, 800, 1000])
for p =1:length(t_)
    subplot(5,2,p);
    boxplot([TableModelsAnalyse{:,14}],TableModelsAnalyse{:,t_{p}})
    title(sprintf('Parameter: %s', t_{p}))
    ylabel('CPU cons. (s)')
end

sgtitle('Train CPU / hyperparameter', 'FontSize', 20);
%%
fig3=figure();
set(gcf, 'Position',  [500, 500, 800, 1000])
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
if CV ==1
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
else
    fprintf('\n Can not do last figure; \n CV should be 1 to check data size impact\n')
end


