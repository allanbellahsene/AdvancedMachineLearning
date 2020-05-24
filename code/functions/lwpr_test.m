function [NMSE, CPU, Y_prediction]= lwpr_test(hyperparameters,X,Y,Xt,Yt,CV)
    global lwprs;
    
    NMSE = zeros(2,length((hyperparameters(:,1))));
    CPU = zeros(2,length((hyperparameters(:,1))));
    Y_prediction = zeros(length(Yt),length((hyperparameters(:,1))));
    
    for k=1:length((hyperparameters(:,1)))
        try
        %% Initialize model
        ID =hyperparameters(k,1); % ID              : desired ID of model
        
        lwpr('Init',ID,hyperparameters(k,2),hyperparameters(k,3),...
            hyperparameters(k,4),hyperparameters(k,5),hyperparameters(k,6),hyperparameters(k,7),hyperparameters(k,8),ones(hyperparameters(k,2),1),[1],'lwpr_test');
        
        lwpr('Change',ID,'init_D',eye(hyperparameters(k,2))*hyperparameters(k,9));
        lwpr('Change',ID,'init_alpha',ones(hyperparameters(k,2))*hyperparameters(k,8));     % this is a safe learning rate
        lwpr('Change',ID,'w_gen',hyperparameters(k,10));                  % more overlap gives smoother surfaces
        lwpr('Change',ID,'init_lambda',hyperparameters(k,11));
        lwpr('Change',ID,'final_lambda',hyperparameters(k,12));
        lwpr('Change',ID,'tau_lambda',hyperparameters(k,13));

        n= length(X);
        
        %% Train Data
        t= cputime; % start time 
        
        inds = randperm(n);
        mse = 0;
        for i=1:n
            [yp,w] = lwpr('Update',ID,X(inds(i),:)',Y(inds(i),:)');
            mse = mse + (Y(inds(i),:)-yp).^2;
        end
        nMSE = mse/n/var(Y,1);
        
        e = cputime-t; % elapsed time 
        
        % store data
        NMSE(1,k)=nMSE;
        CPU(1,k) = e;
        
        % print on the terminal
        fprintf('#ID = %d #Data=%d #rfs=%d nMSE=%5.3f (TrainingSet) CPUtime: %g\n',ID,lwprs(ID).n_data,length(lwprs(ID).rfs),nMSE,e);
        
        %% Prediction
%         
        % create predictions for the test data
        t_2= cputime; % start time
        if CV ==1
            [yp,w,conf]=lwpr('Predict',ID,Xt(1,:)',0.001);
            %
            ep   = Yt-yp;
            mse  = mean(ep.^2);
            nmse = mse/var(Y,1);
            
            e_2 = cputime-t_2; % elapsed time
            
            fprintf('#ID = %d #Data=%d #rfs=%d nMSE=%5.3f (TestSet)  CPUtime: %g\n',ID,lwprs(ID).n_data,length(lwprs(ID).rfs),nmse, e_2);
            
            % store data
            NMSE(2,k)=nmse;
            CPU(2,k)=e_2;
            Y_prediction(1,k)=yp;
        elseif CV ==0
            % create predictions for the test data
            t_2= cputime; % start time
            
            Yp = zeros(size(Yt));
            for i=1:length(Xt),
                [yp,w,conf]=lwpr('Predict',ID,Xt(i,:)',0.001);
                Yp(i,1) = yp;
            end
            ep   = Yt-Yp;
            mse  = mean(ep.^2);
            nmse = mse/var(Y,1);
            
            e_2 = cputime-t_2; % elapsed time
            
            fprintf('#ID = %d #Data=%d #rfs=%d nMSE=%5.3f (TestSet)  CPUtime: %g\n',ID,lwprs(ID).n_data,length(lwprs(ID).rfs),nmse, e_2);
            
            % store data
            NMSE(2,k)=nmse;
            CPU(2,k)=e_2;
            Y_prediction(:,k)=Yp;
        else 
            fprint('Error');
        end
        catch ME
        fprintf('No success: %s\n', ME.message);
        end
    end
end