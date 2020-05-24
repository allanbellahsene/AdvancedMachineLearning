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
        [NMSE_3D(:,:,j), CPU_3D(:,:,j), Y_prediction(j,:)]= lwpr_test(hyperparameters,X,Y,Xt,Yt,CV);
    end
elseif CV == 0
    
    % Train Data
    X = table2array(data(1:n,3:end));
    Y = data.y_t(1:n);
    % Test Data
    Xt = table2array(data(n+1:end,3:end));
    Yt = data.y_t(n+1:end);
    
    [NMSE_3D, CPU_3D, Y_prediction]= lwpr_test(hyperparameters,X,Y,Xt,Yt,CV);
else
    fprint('Error')
end


end

