%{
Function that runs one single prediction step for the Kalman Filter algorithm

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


function [Pred] = prediction_step_KF(Par, Pred, hour, Par_hour, Par_type, Prev_hour, x)

% Parameter B and beta are extracted
% If we are working with the intercept for the emission distribution
% (i.e. beta is present), then B doesn't consider Theta's first row
if Par.EmissionY.intercept
    B = Par.EmissionY.Theta(2:end,:,Par_hour,Par_type)';
    beta = Par.EmissionY.Theta(1,:,Par_hour,Par_type)';
else
    B = Par.EmissionY.Theta(:,:,Par_hour,Par_type)';
    beta = 0;
end

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
Cov1 = inv(Par.Transition.Gamma(:,:,Par_hour,Par_type) + A*Pred.Sigma(:,:,Prev_hour)*A');
Pred.Sigma(:,:,hour) = inv(Cov1 + B'/Par.EmissionY.Gamma(:,:,Par_hour,Par_type)*B);
Pred.Mu(:,hour) = Pred.Sigma(:,:,hour)*(Cov1*Par.Transition.Theta(:,:,Par_hour,Par_type)'*mu + B'/Par.EmissionY.Gamma(:,:,Par_hour,Par_type)*(x-beta));

end