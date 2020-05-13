%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %   Locally Weighted Projected Regression 2D Example  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             1) Test 1D LWPR Example                 %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;
Y = csvread('Fidelity_HighYieldBond_Fund_JPY.csv') ;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             2) Test 2D LWPR Example                 %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;
test_lwpr_2D();

%%
% go to folders up the hierarchy:
upUpFolder = fileparts(fileparts(pwd));

s=fullfile('..', 'data', 'FUNDS', 'Fidelity_HighYieldBond_Fund_JPY.csv');
M=xlsread(s);