function [accu,lb,predict_value] = LinearMultiSVM(Xtrain, Ytrain, Xtest, Ytest, C)
addpath('/home/tzeng/liblinear-1.96/matlab');

% Ytrain(Ytrain~=1) = -1;
% Ytest(Ytest~=1) = -1;
% if size(Ytrain,1)==1
    % Ytrain = Ytrain';
% end
if size(Ytest,1)==1
    Ytest = Ytest';
end

% libsvm
%param1 = sprintf('-b 1 -t 0 -c %f',C);
%model = train(Ytrain, Xtrain, param1);
%[pred_label, accuracy, pred_value] = predict(Ytest, Xtest, model, '-b 1');

% liblinear
trainSize = size(Xtrain);
instanceN =trainSize(1);
featureN  =trainSize(2);

if instanceN<featureN
	param1 = sprintf('-s 7  -c %f -q',C); % s=7 : L2-regularized logistic regression (dual), when instance << features
else
	param1 = sprintf('-s 0  -c %f -q',C); % s=0 : L2-regularized logistic regression (primal), when instance >> features
end

%disp('Lib SVM start train...')
model = train(Ytrain, sparse(Xtrain), param1);
%disp('Lib SVM finished train...')
[lb, accu, prob_estimates] = predict(Ytest, sparse(Xtest), model, '-b 1');

predict_value = prob_estimates;

%if size(model.Label,2)>1
    % if model.Label(1)==1
        % predict_value = prob_estimates(:,1);
    % else
        % predict_value = prob_estimates(:,2);
    % end
%else
%    predict_value = prob_estimates;
%end
%auc=0;

if length(predict_value) ~=length(Ytest)
 disp(['there are ' num2str(length(predict_value) ) ' of predict labels and ' num2str(length(Ytest)) ' of Ytest labels']);
end
%auc = roc(predict_value,Ytest,'nofigure');
%disp(['auc = ' num2str(auc)]);
%auc = roc(lb,Ytest,'nofigure');

