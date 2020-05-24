function [NMSE_3D,CPU_3D,Y_prediction] = lwpr_run(hyperparameters,data,CV,NMSE_3D,CPU_3D,Y_prediction)
%LWPR_RUN Summary of this function goes here
%   Detailed explanation goes here
n= round(height(data)*0.7);
nn = height(data)-n;
if CV ==1
    for j = 1:nn
        % Train Data
        X = table2array(data(1:n+j-1,3:end));
        Y = data.y_t(1:n+j-1);
        % Initial Test
        Xt = table2array(data(n+j:n+j,3:end));
        Yt = data.y_t(n+j:n+j);
        [NMSE_3D(:,:,j), CPU_3D(:,:,j), Y_prediction(j,:)]= lwpr_train_predict(hyperparameters,X,Y,Xt,Yt,CV);
    end
elseif CV == 0
    
    % Train Data
    X = table2array(data(1:n,3:end));
    Y = data.y_t(1:n);
    % Test Data
    Xt = table2array(data(n+1:end,3:end));
    Yt = data.y_t(n+1:end);
    
    [NMSE_3D, CPU_3D, Y_prediction]= lwpr_train_predict(hyperparameters,X,Y,Xt,Yt,CV);
elseif CV == 2
    % Total Train Data
    mm=size(table2array(data(1:n,3:end)),2);

    for m = 1:mm-3
        % Train Data
        X = table2array(data(1:n,3:3+m));
        Y = data.y_t(1:n);
        % Initial Test
        Xt = table2array(data(n+1:end,3:3+m));
        Yt = data.y_t(n+1:end);
        [NMSE_3D(:,:,m), CPU_3D(:,:,m), Y_prediction(:,:,m)]= lwpr_train_predict(hyperparameters,X,Y,Xt,Yt,CV);
    end
else
    fprint('Error')
end


end

