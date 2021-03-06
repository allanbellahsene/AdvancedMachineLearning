%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=======================================================================================================%
%====================================== Advanced Machine Learning ======================================% 
%========================================= Team O - SVR vs LWPR ========================================%
%==================================== BRODARD Lionel, BELLAHSENE Allan =================================%
%========================================= Data - Pre-processing =======================================%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;

%% FUNDS IMPORTS

% Funds  List
funds = {'CreditSuisse_Commodity_US.csv','CreditSuisse_Fund_Bond_CHF.csv',...
    'Fidelity_HighYieldBond_Fund_JPY.csv','Pictet_Gold_Fund_Japan.csv','Vanguard500_Equity_Fund_US.csv'};


opts = delimitedTextImportOptions("NumVariables", 6);
% Specify range and delimiter
opts.DataLines = [2,1300];
opts.Delimiter = ",";
% Specify column names and types
opts.VariableNames = ["Date", "Price", "Open", "High", "Low", "Change"];
opts.VariableTypes = ["datetime", "double", "double", "double", "double", "double"];
% Specify file level properties
opts.ImportErrorRule = "error";
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
% Specify variable properties
opts = setvaropts(opts, "Date", "InputFormat", "MMM d, yyyy");
opts = setvaropts(opts, ["Price", "Open", "High", "Low", "Change"], "TrimNonNumeric", true);
opts = setvaropts(opts, ["Price", "Open", "High", "Low", "Change"], "ThousandsSeparator", ",");

% Path of funds
for i=1:5
    path(i)= fullfile('..', 'data', 'FUNDS', funds(i));
end

% Import of funds
CS_Commo_US= readtable(path{1}, opts);
CS_Commo_US= CS_Commo_US(:,[1,2]);
CS_Bond_CHF= readtable(path{2}, opts);
CS_Bond_CHF= CS_Bond_CHF(:,[1,2]);
Fidelity_Bond_JPY= readtable(path{3}, opts);
Fidelity_Bond_JPY= Fidelity_Bond_JPY(:,[1,2]);
PictetGold_JPY= readtable(path{4}, opts);
PictetGold_JPY= PictetGold_JPY(:,[1,2]);
Vanguard_Equity_US= readtable(path{5}, opts);
Vanguard_Equity_US= Vanguard_Equity_US(:,[1,2]);

%Clear temporary variables
clear opts

% Join Funds
Funds_Prices = innerjoin(CS_Commo_US, CS_Bond_CHF,'LeftKeys',1,'RightKeys',1);
Funds_Prices = innerjoin(Funds_Prices, Fidelity_Bond_JPY,'LeftKeys',1,'RightKeys',1);
Funds_Prices = innerjoin(Funds_Prices, PictetGold_JPY,'LeftKeys',1,'RightKeys',1);
Funds_Prices = innerjoin(Funds_Prices, Vanguard_Equity_US,'LeftKeys',1,'RightKeys',1);
Funds_Prices=rmmissing(Funds_Prices);

% Compute return of funds
Funds_Returns(:,1)=Funds_Prices(2:end,1);  
for i = 2:6
    Funds_Returns(:,i)=array2table(log(table2array(Funds_Prices(2:end,i))./table2array(Funds_Prices(1:end-1,i))));
end

% Change date format
Funds_Returns.Date=datetime(Funds_Returns.Date,'Format','yyyy-MM-dd');
Funds_Returns.Properties.VariableNames={'Date' 'CS_Commo' 'CS_Bond' 'Fidelity_Bond' 'Pictet_Gold' 'Vanguard_Equity'};

% Export data
writetable(Funds_Returns,fullfile('..', 'data', 'CLEANED','Funds_Returns.dat'),'WriteRowNames',true)  

%% YIELDS IMPORTS

yields = {'US10Y_data.csv'};

% Setup the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 6);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["Date", "Price", "Var3", "Var4", "Var5", "Var6"];
opts.SelectedVariableNames = ["Date", "Price"];
opts.VariableTypes = ["datetime", "double", "string", "string", "string", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["Var3", "Var4", "Var5", "Var6"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["Var3", "Var4", "Var5", "Var6"], "EmptyFieldRule", "auto");
opts = setvaropts(opts, "Date", "InputFormat", "");
for i=1:1
    path(i)= fullfile('..', 'data', 'FACTORS','TREASURY_YIELDS', yields(i));
end

% Import the data
US10Y = readtable(path{1}, opts);
US10Y.Date=datetime(US10Y.Date,'Format','yyyy-MM-dd');

% Clear temporary variables
clear opts


% Join Yields

Yields=rmmissing(US10Y);

% divide by 100 add 1
Yields(:,[2])=array2table(table2array(Yields(:,[2]))/100+1);


% Compute return of funds
delta_Yields(:,1)=Yields(2:end,1);  

delta_Yields(:,2)=array2table(log(table2array(Yields(2:end,2))./table2array(Yields(1:end-1,2))));

delta_Yields.Properties.VariableNames={'Date' 'US10Y'}; 

% Export data
writetable(delta_Yields,fullfile('..', 'data', 'CLEANED','delta_Yields.dat'),'WriteRowNames',true)  

%% YIELDS IMPORTS

yields = {'Switzerland10Y.csv','Japan10Y.csv' ,'Germany10Y.csv', 'France10Y.csv','US10Y_data.csv'};

% Setup the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 6);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["Date", "Price", "Var3", "Var4", "Var5", "Var6"];
opts.SelectedVariableNames = ["Date", "Price"];
opts.VariableTypes = ["datetime", "double", "string", "string", "string", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["Var3", "Var4", "Var5", "Var6"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["Var3", "Var4", "Var5", "Var6"], "EmptyFieldRule", "auto");
opts = setvaropts(opts, "Date", "InputFormat", "");
for i=1:5
    path(i)= fullfile('..', 'data', 'FACTORS','TREASURY_YIELDS', yields(i));
end

% Import the data
Switzerland10Y = readtable(path{1}, opts);
Switzerland10Y.Date=datetime(Switzerland10Y.Date,'Format','yyyy-MM-dd');
Japan10Y = readtable(path{2}, opts);
Japan10Y.Date=datetime(Japan10Y.Date,'Format','yyyy-MM-dd');
Germany10Y = readtable(path{3}, opts);
Germany10Y.Date=datetime(Germany10Y.Date,'Format','yyyy-MM-dd');
France10Y = readtable(path{4}, opts);
France10Y.Date=datetime(France10Y.Date,'Format','yyyy-MM-dd');
US10Y = readtable(path{5}, opts);
US10Y.Date=datetime(US10Y.Date,'Format','yyyy-MM-dd');

% Clear temporary variables
clear opts


% Join Yields
Yields = innerjoin(Switzerland10Y, Japan10Y,'LeftKeys',1,'RightKeys',1);
Yields = innerjoin(Yields, Germany10Y,'LeftKeys',1,'RightKeys',1);
Yields = innerjoin(Yields, France10Y,'LeftKeys',1,'RightKeys',1);
Yields = innerjoin(Yields, US10Y,'LeftKeys',1,'RightKeys',1);
Yields=rmmissing(Yields);

% divide by 100 add 1
Yields(:,[2:6])=array2table(table2array(Yields(:,[2:6]))/100+1);


% Compute return of funds
delta_Yields(:,1)=Yields(2:end,1);  
for i = 2:6
    delta_Yields(:,i)=array2table(log(table2array(Yields(2:end,i))./table2array(Yields(1:end-1,i))));
end
delta_Yields.Properties.VariableNames={'Date' 'Switzerland10Y' 'Japan10Y' 'Germany10Y' 'France10Y' 'US10Y'}; 

% Export data
writetable(delta_Yields,fullfile('..', 'data', 'CLEANED','delta_Yields.dat'),'WriteRowNames',true)  

%% MARKET INDICES IMPORTS

indices = {'CAC40_data.csv','DAX_data.csv' ,'DOWJONES_data.csv', 'EURO100_data.csv','NIKKEI225_data.csv','SP500_data.csv','VIX_data.csv'};

% Setup the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 7);

% Specify range and delimiter
opts.DataLines = [1, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["Date", "Var2", "Var3", "Var4", "Var5", "AdjClose", "Var7"];
opts.SelectedVariableNames = ["Date", "AdjClose"];
opts.VariableTypes = ["datetime", "string", "string", "string", "string", "double", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["Var2", "Var3", "Var4", "Var5", "Var7"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["Var2", "Var3", "Var4", "Var5", "Var7"], "EmptyFieldRule", "auto");
opts = setvaropts(opts, "Date", "InputFormat", "yyyy-MM-dd");

% Import the data

% Path
for i=1:7
    path(i)= fullfile('..', 'data', 'FACTORS','MARKET INDICES', indices(i));
end

% Import the data
CAC40_data = readtable(path{1}, opts);
CAC40_data.Date=datetime(CAC40_data.Date,'Format','yyyy-MM-dd');
DAX_data = readtable(path{2}, opts);
DAX_data.Date=datetime(DAX_data.Date,'Format','yyyy-MM-dd');
DOWJONES_data = readtable(path{3}, opts);
DOWJONES_data.Date=datetime(DOWJONES_data.Date,'Format','yyyy-MM-dd');
EURO100_data = readtable(path{4}, opts);
EURO100_data.Date=datetime(EURO100_data.Date,'Format','yyyy-MM-dd');
NIKKEI225_data = readtable(path{5}, opts);
NIKKEI225_data.Date=datetime(NIKKEI225_data.Date,'Format','yyyy-MM-dd');
SP500_data = readtable(path{5}, opts);
SP500_data.Date=datetime(SP500_data.Date,'Format','yyyy-MM-dd');
VIX_data = readtable(path{5}, opts);
VIX_data.Date=datetime(VIX_data.Date,'Format','yyyy-MM-dd');


% Join Indices
marketIndices= innerjoin(CAC40_data, DAX_data,'LeftKeys',1,'RightKeys',1);
marketIndices = innerjoin(marketIndices, DOWJONES_data,'LeftKeys',1,'RightKeys',1);
marketIndices = innerjoin(marketIndices, EURO100_data,'LeftKeys',1,'RightKeys',1);
marketIndices = innerjoin(marketIndices, NIKKEI225_data,'LeftKeys',1,'RightKeys',1);
marketIndices = innerjoin(marketIndices, SP500_data,'LeftKeys',1,'RightKeys',1);
marketIndices = innerjoin(marketIndices, VIX_data,'LeftKeys',1,'RightKeys',1);
marketIndices=rmmissing(marketIndices);
% Compute return of funds
marketIndices_Returns(:,1)=marketIndices(2:end,1);  
for i = 2:8
    marketIndices_Returns(:,i)=array2table(log(table2array(marketIndices(2:end,i))./table2array(marketIndices(1:end-1,i))));
end
marketIndices_Returns.Properties.VariableNames={'Date' 'CAC40' 'DAX' 'DOWJONES' 'EURO100' 'NIKKEI225' 'SP500' 'VIX'};
% Export data
writetable(marketIndices_Returns,fullfile('..', 'data', 'CLEANED','marketIndices_Returns.dat'),'WriteRowNames',true)  

%% EXCHANGE RATES IMPORTS

xRates = {'EURCHF.csv','EURGBP.csv' ,'EURUSD.csv', 'GBPUSD.csv','USDJPY.csv'};

% Import the data

% Path
for i=1:5
    path(i)= fullfile('..', 'data', 'FACTORS','EXCHANGE RATES', xRates(i));
end

% Import the data
EURCHF = readtable(path{1}, opts);
EURCHF.Date=datetime(EURCHF.Date,'Format','yyyy-MM-dd');
EURGBP = readtable(path{2}, opts);
EURGBP.Date=datetime(EURGBP.Date,'Format','yyyy-MM-dd');
EURUSD = readtable(path{3}, opts);
EURUSD.Date=datetime(EURUSD.Date,'Format','yyyy-MM-dd');
GBPUSD = readtable(path{4}, opts);
GBPUSD.Date=datetime(GBPUSD.Date,'Format','yyyy-MM-dd');
USDJPY = readtable(path{5}, opts);
USDJPY.Date=datetime(USDJPY.Date,'Format','yyyy-MM-dd');

% Join Yields
exchangeRates = innerjoin(EURCHF, EURGBP,'LeftKeys',1,'RightKeys',1);
exchangeRates = innerjoin(exchangeRates, EURUSD,'LeftKeys',1,'RightKeys',1);
exchangeRates = innerjoin(exchangeRates, GBPUSD,'LeftKeys',1,'RightKeys',1);
exchangeRates = innerjoin(exchangeRates, USDJPY,'LeftKeys',1,'RightKeys',1);

exchangeRates=rmmissing(exchangeRates);
% Compute return of funds
exchangeRates_Changes(:,1)=exchangeRates(2:end,1);  
for i = 2:6
    exchangeRates_Changes(:,i)=array2table(log(table2array (exchangeRates(2:end,i))./table2array(exchangeRates(1:end-1,i))));
end
exchangeRates_Changes.Properties.VariableNames={'Date' 'EURCHF' 'EURGBP'  'EURUSD'  'GBPUSD' 'USDJPY'};
% Export data
writetable(exchangeRates_Changes,fullfile('..', 'data', 'CLEANED','exchangeRates_Changes.dat'),'WriteRowNames',true)  

%% COMMODITIES IMPORTS

commodities = {'GOLD_data.csv','OIL_data.csv' ,'SILVER_data.csv'};

% Import the data

% Path
for i=1:3
    path(i)= fullfile('..', 'data', 'FACTORS','COMMODITIES', commodities(i));
end

% Import the data
GOLD_data = readtable(path{1}, opts);
EURCGOLD_dataHF.Date=datetime(GOLD_data.Date,'Format','yyyy-MM-dd');
OIL_data = readtable(path{2}, opts);
OIL_data.Date=datetime(OIL_data.Date,'Format','yyyy-MM-dd');
SILVER_data = readtable(path{3}, opts);
SILVER_data.Date=datetime(SILVER_data.Date,'Format','yyyy-MM-dd');


% Join Yields
commodities_Price = innerjoin(GOLD_data, OIL_data,'LeftKeys',1,'RightKeys',1);
commodities_Price = innerjoin(commodities_Price, SILVER_data,'LeftKeys',1,'RightKeys',1);
commodities_Price=rmmissing(commodities_Price);
commodities_Price.Date=datetime(commodities_Price.Date,'Format','yyyy-MM-dd');
% Compute return of funds
commodities_Return(:,1)=commodities_Price(2:end,1);  
for i = 2:4
    commodities_Return(:,i)=array2table(log(table2array (commodities_Price(2:end,i))./table2array(commodities_Price(1:end-1,i))));
end
commodities_Return.Properties.VariableNames={'Date' 'GOLD' 'OIL'  'SILVER'};
% Export data
writetable(commodities_Return,fullfile('..', 'data', 'CLEANED','commodities_Return.dat'),'WriteRowNames',true)  
