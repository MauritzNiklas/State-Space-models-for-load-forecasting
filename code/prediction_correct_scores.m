%{
Function that simply divides the prediction scores by the number of
prediction days

It requires:
- Pred: structure that contains information about the predictions
- prediction_days: the number of days for which the load is predicted

It outputs:
- Pred: structure that contains the updated score variables
%}


function [Pred] = prediction_correct_scores(Pred, prediction_days)

Pred.MAPE = Pred.MAPE/prediction_days;
Pred.MSE = Pred.MSE/prediction_days;
Pred.LogScore = Pred.LogScore/prediction_days;
Pred.PinballLoss = Pred.PinballLoss/prediction_days;

end