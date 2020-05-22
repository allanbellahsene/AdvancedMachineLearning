%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=======================================================================================================%
%====================================== Advanced Machine Learning ======================================%
%========================================= Team O - SVR vs LWPR ========================================%
%==================================== BRODARD Lionel, BELLAHSENE Allan =================================%
%================================================ main =================================================%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;

addpath(genpath('functions')) % add function apth
rng default % keep same random numbers

%% Imports data
data = readtable(fullfile('..', 'data', 'CLEANED', 'cleaned_data.dat'));

%% d-dimension
n= round(height(data)*0.7);
X = table2array(data(1:n,3:end));
Y = data.y_t(1:n);
Xt = table2array(data(n+1:end,3:end));
Yt = data.y_t(n+1:end);
%% initialize LWPR
n_in = size(X,2) ; % : number of input dimensions
n_out = size(Y,2) ; % : number of output dimensions

% diag_only = [0  1]; % : 1/0 to update only the diagonal distance metric
% meta = [0 1]; % : 1/0 to allow the use of a meta learning parameter
% meta_rate = [1 10 100 1000]; % : the meta learning rate
% penalty = [1.e-4 1.e-5 1.e-6 1.e-7]; % : a smoothness bias, usually a pretty small number (1.e-4)
% init_alpha= [1 10 100 1000]; % : the initial learning rates

diag_only = [0 1]; % : 1/0 to update only the diagonal distance metric
meta = [0 1]; % : 1/0 to allow the use of a meta learning parameter
meta_rate = [1 10 100 1000]; % : the meta learning rate
penalty = [1.e-4 1.e-5 1.e-6 1.e-7]; % : a smoothness bias, usually a pretty small number (1.e-4)
init_alpha= [1 10 100 1000]; % : the initial learning rates
init_D= [1 25 50 200];
w_gen = [0.0001 0.00025 0.0005 0.00075 0.001];
init_lambda = [0.99];
final_lambda = [0.99];
tau_lambda = [0.99];

hyperparameters=transpose(combvec(n_in, n_out, diag_only,meta,meta_rate,penalty, init_alpha, init_D, w_gen, init_lambda, final_lambda, tau_lambda));
IDs = transpose([1:1:length(hyperparameters)]);
hyperparameters = [IDs hyperparameters];

T = array2table(hyperparameters);
T.Properties.VariableNames={'ID' 'n_in' 'n_out' 'diag_only' 'meta' 'meta_rate' 'penalty' 'init_alpha' 'init_D' 'w_gen' 'init_lambda' 'final_lambda' 'tau_lambda'};
%% Hyperparameters to Test
% 

rows = (T.diag_only==1 & T.meta==1 & T.meta_rate==100 & T.init_alpha==100 & T.init_D==25);
hypToTest = T(rows,:);
hyperparameters=table2array(hypToTest);
%%
[NMSE, CPU, Y_prediction]= lwpr_test(table2array(hypToTest),X,Y,Xt,Yt);

%%
[value, index] = min(NMSE(2,:));
[row, col] = ind2sub(size(NMSE), index);
%% Plot
plot(Y_prediction(:,col))
hold on
plot(Yt)
%% OLD

% %% Model
% for k=1:length(hyperparameters(:,1))
%     global lwprs
%     % Initialize model
%     ID =hyperparameters(k,1); % ID              : desired ID of model
%     lwpr('Init',ID,hyperparameters(k,2),hyperparameters(k,3),...
%         hyperparameters(k,4),hyperparameters(k,5),hyperparameters(k,6),hyperparameters(k,7),hyperparameters(k,8),ones(hyperparameters(k,2),1),[1],'lwpr_test');
%     
%     
%     kernel = 'Gaussian';
%     lwpr('Change',ID,'init_D',eye(hyperparameters(k,2))*hyperparameters(k,9));
%     lwpr('Change',ID,'init_alpha',ones(hyperparameters(k,2))*hyperparameters(k,8));     % this is a safe learning rate
%     lwpr('Change',ID,'w_gen',hyperparameters(k,10));                  % more overlap gives smoother surfaces
% %     lwpr('Change',ID,'init_lambda',hyperparameters(k,11));
% %     lwpr('Change',ID,'w_gen',0.2);                  % more overlap gives smoother surfaces
%     lwpr('Change',ID,'init_lambda',0.995);
%     lwpr('Change',ID,'final_lambda',0.9999);
%     lwpr('Change',ID,'tau_lambda',0.9999);
% 
%     % Train Data
%     for j=1:1
%         inds = randperm(n);
%         mse = 0;
%         for i=1:n
%             [yp,w] = lwpr('Update',ID,X(inds(i),:)',Y(inds(i),:)');
%             mse = mse + (Y(inds(i),:)-yp).^2;
%         end
%         nMSE = mse/n/var(Y,1); % std ou var ??
%         disp(sprintf('#ID = %d #Data=%d #rfs=%d nMSE=%5.3f (TrainingSet)',ID,lwprs(ID).n_data,length(lwprs(ID).rfs),nMSE));
%     end
%     NMSE(1,k)=nMSE;
%     % Prediction
%     % create predictions for the test data
%     Yp = zeros(size(Yt));
%     for i=1:length(Xt),
%         [yp,w,conf]=lwpr('Predict',ID,Xt(i,:)',0.001);
%         Yp(i,1) = yp;
%     end
%     ep   = Yt-Yp;
%     mse  = mean(ep.^2);
%     nmse = mse/var(Y,1);
%     disp(sprintf('#ID = %d #Data=%d #rfs=%d nMSE=%5.3f (TestSet)',ID,lwprs(ID).n_data,length(lwprs(ID).rfs),nmse));
%     NMSE(2,k)=nmse;
%     Y_prediction(:,k)=Yp;
% end


% %% Function
% function [NMSE, Y_prediction]= lwpr_test(hyperparameters,X,Y,Xt,Yt)
%     global lwprs;
%     for k=1:length(hyperparameters)
%         % Initialize model
%         ID =hyperparameters(k,1); % ID              : desired ID of model
%         lwpr('Init',ID,hyperparameters(k,2),hyperparameters(k,3),...
%             hyperparameters(k,4),hyperparameters(k,5),hyperparameters(k,6),hyperparameters(k,7),hyperparameters(k,8),ones(hyperparameters(k,2),1),[1],'lwpr_test');
% 
% 
%         kernel = 'Gaussian';
%         lwpr('Change',ID,'init_D',eye(hyperparameters(k,2))*hyperparameters(k,9));
%         lwpr('Change',ID,'init_alpha',ones(hyperparameters(k,2))*hyperparameters(k,8));     % this is a safe learning rate
%         lwpr('Change',ID,'w_gen',hyperparameters(k,10));                  % more overlap gives smoother surfaces
%         %     lwpr('Change',ID,'init_lambda',hyperparameters(k,11));
%         %     lwpr('Change',ID,'w_gen',0.2);                  % more overlap gives smoother surfaces
%         lwpr('Change',ID,'init_lambda',0.995);
%         lwpr('Change',ID,'final_lambda',0.9999);
%         lwpr('Change',ID,'tau_lambda',0.9999);
%         n= length(X);
%         % Train Data
%         for j=1:1
%             inds = randperm(n);
%             mse = 0;
%             for i=1:n
%                 [yp,w] = lwpr('Update',ID,X(inds(i),:)',Y(inds(i),:)');
%                 mse = mse + (Y(inds(i),:)-yp).^2;
%             end
%             nMSE = mse/n/var(Y,1); % std ou var ??
%             disp(sprintf('#ID = %d #Data=%d #rfs=%d nMSE=%5.3f (TrainingSet)',ID,lwprs(ID).n_data,length(lwprs(ID).rfs),nMSE));
%         end
%         NMSE(1,k)=nMSE;
%         % Prediction
%         % create predictions for the test data
%         Yp = zeros(size(Yt));
%         for i=1:length(Xt),
%             [yp,w,conf]=lwpr('Predict',ID,Xt(i,:)',0.001);
%             Yp(i,1) = yp;
%         end
%         ep   = Yt-Yp;
%         mse  = mean(ep.^2);
%         nmse = mse/var(Y,1);
%         disp(sprintf('#ID = %d #Data=%d #rfs=%d nMSE=%5.3f (TestSet)',ID,lwprs(ID).n_data,length(lwprs(ID).rfs),nmse));
%         NMSE(2,k)=nmse;
%         Y_prediction(:,k)=Yp;
%     end
% end