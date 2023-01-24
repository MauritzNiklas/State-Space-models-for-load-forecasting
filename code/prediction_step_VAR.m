%{
Function that runs one single prediction step for the VAR Model algorithm

It requires:
- Par: structure that contains the trained parameters
- Pred: structure that contains information about the predictions
- hour: time parameter used to index the output parameters
- Par_hour: time parameter used for the parameters' index
- Par_type: type of day parameter used for the parameters' index
- Prev_hour: time parameter used to extract the mean and covariance of the previous step

It outputs:
- Pred: structure that contains the updated parameters for mean and covariance
%}


function [Pred] = prediction_step_VAR(Par, Pred, hour, Par_hour, Par_type, Prev_hour)

% Now the previous mean (mu) and parameter A are extracted
% If we are working with the intercept for the transition distribution
% (i.e. alpha is present), then the mean contains a 1 and A doesn't 
% consider Theta's first row
if Par.Transition.intercept
    mu = [1; Pred.Mu(:,Prev_hour)];
    A = Par.Transition.Theta(2:end,:,Par_hour,Par_type)';
else
    mu = Pred.Mu(:,Prev_hour);
    A = Par.Transition.Theta(:,:,Par_hour,Par_type)';
end

% With these formulas, new mean and covariance are calculated
Pred.Mu(:,hour) = Par.Transition.Theta(:,:,Par_hour,Par_type)'*mu;
Pred.Sigma(:,:,hour) = Par.Transition.Gamma(:,:,Par_hour,Par_type) + A*Pred.Sigma(:,:,Prev_hour)*A';

end