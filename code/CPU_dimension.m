clear; close all; clc;

addpath(genpath('functions'))                   % add function apth
rng default                                     % keep same random numbers

%% Imports data
data = readtable(fullfile('..', 'data', 'CLEANED', 'cleaned_data.dat'));

%% initialize LWPR

diagOnly = [1];                              % : 1/0 to update only the diagonal distance metric
meta =  [1];                                   % : 1/0 to allow the use of a meta learning parameter
metaRate = [  0.01];                    % : the meta learning rate
penalty = [ 0.0001];            % : a smoothness bias, usually a pretty small number (1.e-4)
initAlpha= [  0.01 ];                    % : the initial learning rates
initD= [  0.0001  ];                          % : the initial distance metrics
wGen = [ 0.0002];  % : weight
initLambda = [1.e-5];                           % : the initial lambda
finalLambda = [0.99 ];                          % : the final lambda
tauLambda = [  0.1];                            % : the tau lambda

%% Initial Train & Test

% Initial n
n= round(height(data)*0.7);

% Initial Train Data
X = table2array(data(1:n,3:end));
Y = data.y_t(1:n);

% Initial Test
Xt = table2array(data(n+1:n+1,3:end));
Yt = data.y_t(n+1:n+1);

% Create a matrix with all the hyperparameters
hyperparameters = transpose(combvec(diagOnly,meta,metaRate,...
    penalty, initAlpha, initD, wGen, initLambda, finalLambda, tauLambda));


% Set up an ID for each combinaison & add it to the matrix
IDs = transpose([1:1:size(hyperparameters,1)]);
hyperparameters = [IDs hyperparameters];

% Transform the Matrix to a Table
Hyperparameters = array2table(hyperparameters);
Hyperparameters.Properties.VariableNames={'ID' 'diagOnly' 'meta' ...
    'metaRate' 'penalty' 'initAlpha' 'initD' 'wGen' 'initLambda' ...
    'finalLambda' 'tauLambda'};
hyperparameters_table = Hyperparameters(:,:);
hyperparameters_array=table2array(hyperparameters_table);
%% Launch LWPR Algorithm
% Initial n
mm=size(table2array(data(1:n,3:end)),2);
nn = height(data)-n;

NMSE_3D=zeros(2,length((hyperparameters_array(:,1))),mm-2);
CPU_3D=zeros(2,length((hyperparameters_array(:,1))),mm-2);
yPrediction = zeros(nn,1,mm-2);
% Choose the Hyperparameters to run:


CV = 2;

[NMSE_3D,CPU_3D,yPrediction] = lwpr_run(hyperparameters_array,data,CV,NMSE_3D,CPU_3D,yPrediction);

%%
%
CPU_N=zeros(mm-3,1);
for i =1:mm-3
    CPU_N(i,1)= mean(mean(CPU_3D(1,:,i)));
end
dim = transpose([1:1:mm-3]);

CPU_Dim=[dim CPU_N];

fig5=figure();
scatter(CPU_Dim(:,1),CPU_Dim(:,2),'b')
h = lsline;
set(h(1),'color','r','LineWidth',1.2)
set(gca,'FontSize',16)
title('CPU time to train LWPR as function of # of dimensions','FontSize', 20)
xlabel('# of dimensions','FontSize', 16)
ylabel('time in seconds','FontSize', 16)
legend({ 'CPU Time (s)'})
ylim([0.1 0.8])
xlim([1 16])
saveas(fig5,fullfile('..', 'figures','CPU_LWPR_D.pdf'));

