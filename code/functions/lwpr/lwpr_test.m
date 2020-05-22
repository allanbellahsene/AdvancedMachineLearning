function [NMSE, Y_prediction]= lwpr_test(hyperparameters,X,Y,Xt,Yt)
    global lwprs;
    for k=1:length((hyperparameters(:,1)))
        % Initialize model
        ID =hyperparameters(k,1); % ID              : desired ID of model
        lwpr('Init',ID,hyperparameters(k,2),hyperparameters(k,3),...
            hyperparameters(k,4),hyperparameters(k,5),hyperparameters(k,6),hyperparameters(k,7),hyperparameters(k,8),ones(hyperparameters(k,2),1),[1],'lwpr_test');


        kernel = 'Gaussian';
        lwpr('Change',ID,'init_D',eye(hyperparameters(k,2))*hyperparameters(k,9));
        lwpr('Change',ID,'init_alpha',ones(hyperparameters(k,2))*hyperparameters(k,8));     % this is a safe learning rate
        lwpr('Change',ID,'w_gen',hyperparameters(k,10));                  % more overlap gives smoother surfaces
        %     lwpr('Change',ID,'init_lambda',hyperparameters(k,11));
        %     lwpr('Change',ID,'w_gen',0.2);                  % more overlap gives smoother surfaces
        lwpr('Change',ID,'init_lambda',0.995);
        lwpr('Change',ID,'final_lambda',0.9999);
        lwpr('Change',ID,'tau_lambda',0.9999);
        n= length(X);
        % Train Data
        for j=1:1
            inds = randperm(n);
            mse = 0;
            for i=1:n
                [yp,w] = lwpr('Update',ID,X(inds(i),:)',Y(inds(i),:)');
                mse = mse + (Y(inds(i),:)-yp).^2;
            end
            nMSE = mse/n/var(Y,1); % std ou var ??
            disp(sprintf('#ID = %d #Data=%d #rfs=%d nMSE=%5.3f (TrainingSet)',ID,lwprs(ID).n_data,length(lwprs(ID).rfs),nMSE));
        end
        NMSE(1,k)=nMSE;
        % Prediction
        % create predictions for the test data
        Yp = zeros(size(Yt));
        for i=1:length(Xt),
            [yp,w,conf]=lwpr('Predict',ID,Xt(i,:)',0.001);
            Yp(i,1) = yp;
        end
        ep   = Yt-Yp;
        mse  = mean(ep.^2);
        nmse = mse/var(Y,1);
        disp(sprintf('#ID = %d #Data=%d #rfs=%d nMSE=%5.3f (TestSet)',ID,lwprs(ID).n_data,length(lwprs(ID).rfs),nmse));
        NMSE(2,k)=nmse;
        Y_prediction(:,k)=Yp;
    end
end