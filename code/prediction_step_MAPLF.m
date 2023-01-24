%{
Function that runs one single prediction step for the MAPLF algorithm

It requires:
- Par: structure that contains the trained parameters
- Pred: structure that contains information about the predictions
- hour: time parameter used to index the output parameters
- Par_hour: time parameter used for the parameters' index
- Par_type: type of day parameter used for the parameters' index
- Prev_hour: time parameter used to extract the mean and covariance of the previous step
- x: additional observation (e.g. temperature) for the given hour

It outputs:
- Pred: structure that contains the updated parameters for mean and covariance
%}


function [Pred] = prediction_step_MAPLF(Par, Pred, hour, Par_hour, Par_type, Prev_hour, x)

% First, the number of variables in the load matrix (e.g. the number of
% regions) are extracted
n = length(Par.Transition.Cov(:,:,Par_hour,Par_type));

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

% If the emission distibution has the intercept, then a 1 is added
if Par.EmissionX.intercept
    x = [1; x];
end

% With these formulas, new mean and covariance are calculated
Omega = Par.EmissionX.Gamma(:,:,Par_hour,Par_type)/(Par.EmissionX.Gamma(:,:,Par_hour,Par_type) + Par.Transition.Gamma(:,:,Par_hour,Par_type) + A*Pred.Sigma(:,:,Prev_hour)*A');
Pred.Mu(:,hour) = (eye(n) - Omega)*Par.EmissionX.Theta(:,:,Par_hour,Par_type)'*x + Omega*Par.Transition.Theta(:,:,Par_hour,Par_type)'*mu;
Pred.Sigma(:,:,hour) = (eye(n) - Omega)*Par.EmissionX.Gamma(:,:,Par_hour,Par_type);

end