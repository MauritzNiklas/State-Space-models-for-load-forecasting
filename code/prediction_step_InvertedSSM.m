%{
Function that runs one single prediction step for the Inverted State-Space Model algorithm

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


function [Pred] = prediction_step_InvertedSSM(Par, Pred, hour, Par_hour, Par_type, Prev_hour, x)

% First, the number of variables in the observation matrix (m) and in the load matrix (n) are extracted
m = length(x);
n = length(Par.InvertedSSM.Cov(:,:,Par_hour,Par_type)) - m;

% Then the following temporary variables are extracted from Theta and Gamma
A_big = Par.InvertedSSM.Theta(:,1:n,Par_hour,Par_type)';
B_big = Par.InvertedSSM.Theta(:,n+1:end,Par_hour,Par_type)';
Gamma_11 = Par.InvertedSSM.Gamma(1:n,1:n,Par_hour,Par_type);
Gamma_12 = Par.InvertedSSM.Gamma(1:n,n+1:end,Par_hour,Par_type);
Gamma_22 = Par.InvertedSSM.Gamma(n+1:end,n+1:end,Par_hour,Par_type);

% Now the previous mean (mu), A  and B are extracted
% If we are working with the intercept for the transition distribution
% (i.e. alpha and epsilon are present), then the mean contains a 1
if Par.InvertedSSM.intercept
    mu = [1; Pred.Mu(:,Prev_hour)];
    A = A_big(:,2:end);
    B = B_big(:,2:end);
else
    mu = Pred.Mu(:,Prev_hour);
    A = A_big;
    B = B_big;
end

% Definition of other temporary variables
T11 = Gamma_11 + A*Pred.Sigma(:,:,Prev_hour)*A';
T12 = Gamma_12 + A*Pred.Sigma(:,:,Prev_hour)*B';
T22 = Gamma_22 + B*Pred.Sigma(:,:,Prev_hour)*B';

% With these formulas, new mean and covariance are calculated
Pred.Sigma(:,:,hour) = T11 - T12/T22*T12';
Pred.Mu(:,hour) = A_big*mu + T12/T22*(x-B_big*mu);

end