%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=======================================================================================================%
%====================================== Advanced Machine Learning ======================================%
%========================================= Team O - SVR vs LWPR ========================================%
%==================================== BRODARD Lionel, BELLAHSENE Allan =================================%
%================================================ main =================================================%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;
addpath(genpath('functions'))

%% Imports data
data = readtable(fullfile('..', 'data', 'CLEANED', 'cleaned_data.dat'));

%% 2D Silver + Dow Jones
n= round(height(data)*0.75);
X = [data.DOWJONES(1:n) data.SILVER(1:n)];
Y = data.y_t(1:n);
Xt = [data.DOWJONES(n+1:end) data.SILVER(n+1:end)];
Yt = data.y_t(n+1:end);

%% initialize LWPR
global lwprs;

ID = 1;
if ~exist('s') | isempty(s)
    lwpr('Init',ID,2,1,0,0,0,1e-7,50,[1;1],[1],'lwpr_test');
else
    lwpr('Init',ID,s);
end

%% set some parameters
kernel = 'Gaussian';
% kernel = 'BiSquare'; % note: the BiSquare kernel requires different values for
%                              an initial distance metric, as in the next line
% lwpr('Change',ID,'init_D',[7 0.01; 0.01 7]);

lwpr('Change',ID,'init_D',eye(2)*25);
lwpr('Change',ID,'init_alpha',ones(2)*50); % this is a safe learning rate
lwpr('Change',ID,'w_gen',0.2);             % more overlap gives smoother surfaces
lwpr('Change',ID,'meta',1);                % meta learning can be faster, but numerical more dangerous
lwpr('Change',ID,'meta_rate',50);
lwpr('Change',ID,'init_lambda',0.995);
lwpr('Change',ID,'final_lambda',0.9999);
lwpr('Change',ID,'tau_lambda',0.9999);

%% Train
% train the model
for j=1:1
    inds = randperm(n);
    mse = 0;
    for i=1:n
        [yp,w] = lwpr('Update',ID,X(inds(i),:)',Y(inds(i),:)');
        mse = mse + (Y(inds(i),:)-yp).^2;
    end
    nMSE = mse/n/var(Y,1);
    disp(sprintf('#Data=%d #rfs=%d nMSE=%5.3f (TrainingSet)',lwprs(ID).n_data,length(lwprs(ID).rfs),nMSE));
end

%% Prediction
% create predictions for the test data
Yp = zeros(size(Yt));
for i=1:length(Xt),
    [yp,w,conf]=lwpr('Predict',ID,Xt(i,:)',0.001);
    Yp(i,1) = yp;
end
ep   = Yt-Yp;
mse  = mean(ep.^2);
nmse = mse/var(Y,1);
disp(sprintf('#Data=%d #rfs=%d nMSE=%5.3f (TestSet)',lwprs(ID).n_data,length(lwprs(ID).rfs),nmse));

%%
% get the data structure
s = lwpr('Structure',ID);

%% Plot
plot(