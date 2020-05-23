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
yields = readtable(fullfile('..', 'data', 'CLEANED', 'delta_Yields.dat'));
indices_Returns = readtable(fullfile('..', 'data', 'CLEANED', 'marketIndices_Returns.dat'));
xrate_Changes = readtable(fullfile('..', 'data', 'CLEANED', 'exchangeRates_Changes.dat'));

%% Create Equal Weighted Portfolio

portfolio_Return(:,1)=funds_Returns(:,1); 
portfolio_Return(:,2)=array2table(1/5*(table2array(funds_Returns(:,2))+table2array(funds_Returns(:,3)) +table2array(funds_Returns(:,4)) +table2array(funds_Returns(:,5)) +table2array(funds_Returns(:,6)) ));
portfolio_Return(2:end,3)=portfolio_Return(1:end-1,2);
portfolio_Return(3:end,4)=portfolio_Return(1:end-2,2);
portfolio_Return.Properties.VariableNames={'Date' 'y_t' 'y_t_1' 'y_t_2' };

%%
% Create Matrix with t-1
data = innerjoin(portfolio_Return, yields,'LeftKeys',1,'RightKeys',1);
data = innerjoin(data, indices_Returns,'LeftKeys',1,'RightKeys',1);
data = innerjoin(data, commodities_Returns,'LeftKeys',1,'RightKeys',1);
data = innerjoin(data, xrate_Changes,'LeftKeys',1,'RightKeys',1);
data(2:end,5:end)=data(1:end-1,5:end);
data(1:2,:) = [];

columns=data.Properties.VariableNames;


writetable(data,fullfile('..', 'data', 'CLEANED','cleaned_data.dat'),'WriteRowNames',true)  
%% Data visualisation

fig1=figure();

set(gca,'FontSize',16,'XTickLabelRotation',90)
set(gcf, 'Position',  [500, 500, 800, 1000])
for i=1:size(columns,2)-1
    subplot(7,3,i)
    plot(table2array(data(:,1)), table2array(data(:,i+1))) 
    title(sprintf('%s', columns{i+1}))
end
sgtitle('Daily Returns', 'FontSize', 20);
saveas(fig1,fullfile('..', 'figures','daily_return.png'));

%% Functions
dataExploratory= removevars(data,{'Date'});


correlationTable=array2table(corr(dataExploratory.Variables));
correlationTable.Properties.VariableNames=dataExploratory.Properties.VariableNames;
correlationTable.Properties.RowNames=dataExploratory.Properties.VariableNames;
fig3=figure();
set(gcf, 'Position',  [500, 500, 800, 1000])
heatmap(correlationTable.Properties.VariableNames,correlationTable.Properties.VariableNames,abs(correlationTable.Variables),'FontSize',16)
saveas(fig3,fullfile('..', 'figures','Heatmap_01.png'));

%% Boxplot


fig2=figure();
boxplot(dataExploratory.Variables*100,dataExploratory.Properties.VariableNames)
set(gca,'FontSize',16,'XTickLabelRotation',90)
set(gcf, 'Position',  [500, 500, 800, 1000])
ylabel('Value (%)')
title('Boxplot of all variables', 'FontSize', 20)
saveas(fig2,fullfile('..', 'figures','Boxplot_01.png'));
