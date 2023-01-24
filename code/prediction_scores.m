%{
Function that computes all the accuracy metrics for a certain day's
predictions

It requires:
- Pred: structure that contains information about the predictions
- True_loads: variable containing the true loads for the given day
- qs: vector containing the quantiles for which to compute the pinball loss
- day: day in which the prediction occurs (for printing errors)

It outputs:
- Pred: structure that contains the updated score variables
%}


function [Pred] = prediction_scores(Pred, True_load, qs, day)

% The horizon is extracted from the True_loads variable
[horizon, n] = size(True_load);

% MAPE score for the day is calculated and added to the variable
Pred.MAPE = Pred.MAPE + 100*abs(True_load - Pred.Predicted_loads)./True_load;

% MSE score for the day is calculated and added to the variable
Pred.MSE = Pred.MSE + (True_load - Pred.Predicted_loads).^2;

% Pinball loss for the day is calculated and added to the variable
for q=1:length(qs)
    for h = 1:horizon
        for k=1:n
            f = norminv(qs(q),Pred.Predicted_loads(h,k),Pred.Predicted_cov(k,k,h));
            if True_load(h,k)<f
                Pred.PinballLoss(h,k,q) = Pred.PinballLoss(h,k,q) + (1-qs(q))*(f-True_load(h,k));
            else
                Pred.PinballLoss(h,k,q) = Pred.PinballLoss(h,k,q) + qs(q)*(True_load(h,k)-f);
            end
        end
    end
end

% Logarithmic score for the day is calculated and added to the variable
% If an error occurs (it's usually rare) skip the day and print an error
% message
try 
    ls = - log(mvnpdf(True_load, Pred.Predicted_loads, (Pred.Predicted_cov+pagetranspose(Pred.Predicted_cov))/2));
catch
    ls=zeros(horizon,1);
    disp(['Error in LS! in day ' num2str(day)])
end
% If the logarithmic score is infinite for any time period, do not add it
% to the to the variable containing the total score
if any(isinf(ls))
    disp(['Inf! in day ' num2str(day)])
else 
    Pred.LogScore = Pred.LogScore + ls;
end

% The variables containing the total errors, the total predictions and the
% total predictions covariances are updated
Pred.Errors(horizon*(day-1)+(1:horizon),:) = True_load - Pred.Predicted_loads;
Pred.Total_predictions(horizon*(day-1)+(1:horizon),:) = Pred.Predicted_loads;
Pred.Total_predictions_covs(:,:,horizon*(day-1)+(1:horizon)) = Pred.Predicted_cov;

end