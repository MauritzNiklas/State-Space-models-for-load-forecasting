%{
Function that initialises the required parameters for prediction

It requires:
- n: the number of variables in the load matrix (e.g. the number of regions)
- horizon: the length (in hours) of the prediction horizon
- prediction_days: the number of days for which we would like to predict the load
- n_pinball: number of quantiles that are used to compute the Pinball loss

It outputs:
- Pred: structure that contains all the necessary informations about the predictions
%}


function [Pred] = prediction_initialization(n, horizon, prediction_days, n_pinball)

% First, the variables that will contain the mean (Mu) and covariance (Sigma) for the daily prediction phase are created
Pred.Mu = zeros(n, horizon);
Pred.Sigma = zeros(n, n, horizon);

% Then, a copy for the above variables is made, that are essential when creating one-hour-ahead predictions
Pred.Predicted_loads = zeros(horizon, n);
Pred.Predicted_cov = zeros(n, n, horizon);

% Then, the variables that contain the metrics for the daily prediction phase are created. The metrics are:
% - MAPE
% - MSE (to compute the RMSE)
% - Logarithmic score
% - Pinball loss
Pred.MAPE = zeros(horizon, n);
Pred.MSE = zeros(horizon, n);
Pred.LogScore = zeros(horizon, 1);
Pred.PinballLoss = zeros(horizon, n, n_pinball);

% Finally, the variables that contain the prediction errors, all the predictions and all predictions' covariances for all prediction days are created
Pred.Errors = zeros(prediction_days*horizon, n);
Pred.Total_predictions = zeros(prediction_days*horizon, n);
Pred.Total_predictions_covs = zeros(n, n, prediction_days*horizon);

end