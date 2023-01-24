%{
Function that runs all the recursive prediction procedure
In particular, for each day, the load is first predicted, and then the parameters are updated according to the real observations

It requires:
- Par: structure that contains the parameters trained during the training procedure
- loads: the matrix that contains the loads
- observations: the matrix that contains the additional observations (e.g. the temperature)
- times: the matrix that contains the time variables (e.g. hour and day type)
- prediction_days: the number of days for which the load is predicted
- horizon: the length (in hours) of the prediction horizon (can be greater than one day)
- starting_hour: the hour from which the prediction will start. The first observation will be the real load during that time
- one_hour_ahead: boolean value, that states whether to compute one-hour-ahead predictions (i.e. the true load of the previous hour is known) or not
- day_length: the number of observations contained in a day (e.g. for hourly observations, it's 24)
- index: the row index of the loads matrix at the start of the prediction procedure

It outputs:
- Pred: structure that contains all the necessary informations about the predictions
- Par: structure that contains the newly trained parameters
- index: the row index of the loads matrix at the end of the prediction procedure

This function uses a while loop instead of a for loop to update the parameters because for daylight saving time, there are certain days which may contain one hour more or one hour less
%}


function [Pred, Par, index] = prediction(Par, loads, observations, times, prediction_days, horizon, starting_hour, one_hour_ahead, day_length, index)

% Extract the number of variables in the load matrix (e.g. the number of regions)
[~, number_of_classes] = size(loads);

% First, the quantiles that are used to compute the Pinball loss are chosen
qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

% Then the structures that will keep all predictions' information for each forecasting algorithm are initialised, which are:
% - MAPLF
% - Kalman Filter
% - Inverted SSM
% - VAR
Pred.MAPLF = prediction_initialization(number_of_classes, horizon, prediction_days, length(qs));
Pred.KF = prediction_initialization(number_of_classes, horizon, prediction_days, length(qs));
Pred.InvertedSSM = prediction_initialization(number_of_classes, horizon, prediction_days, length(qs));
Pred.VAR = prediction_initialization(number_of_classes, horizon, prediction_days, length(qs));

% Then a variable containing all true loads is created
Pred.True_loads = zeros(prediction_days*horizon, number_of_classes);


% Now the forecasting procedure starts
day=1;
while day<prediction_days+1

    % First the type of day of index+1 (index is the index of the last observation used to train the parameters) is extracted
    day_type = times(index+1,2);
    % Then the parameters are trained until the starting hour for the predictions is reached (if starting_hour is 1, no training will occur)
    for i=1:starting_hour-1
        index = index+1;
        hour = times(index,1);
        Par.Transition = train_step(loads(index,:)', loads(index-1,:)', Par.Transition, hour, day_type, true);
        Par.EmissionX = train_step(loads(index,:)', observations(index,:)', Par.EmissionX, hour, day_type, true);
        Par.EmissionY = train_step(observations(index,:)', loads(index,:)', Par.EmissionY, hour, day_type, true);
        Par.InvertedSSM = train_step([loads(index,:)'; observations(index,:)'], loads(index-1,:)', Par.InvertedSSM, hour, day_type, true);
    end

    % Then, the mean (Mu) and the covariance (Sigma) parameters at the last time index are initialised for each prediction method
    Pred.MAPLF.Sigma(:,:,horizon) = zeros(number_of_classes, number_of_classes);
    Pred.MAPLF.Mu(:,horizon) = loads(index,:)';
    Pred.KF.Sigma(:,:,horizon) = zeros(number_of_classes, number_of_classes);
    Pred.KF.Mu(:,horizon) = loads(index,:)';
    Pred.InvertedSSM.Sigma(:,:,horizon) = zeros(number_of_classes, number_of_classes);
    Pred.InvertedSSM.Mu(:,horizon) = loads(index,:)';
    Pred.VAR.Sigma(:,:,horizon) = zeros(number_of_classes, number_of_classes);
    Pred.VAR.Mu(:,horizon) = loads(index,:)';

    % Then the previous hour parameter is defined
    Prev_hour = horizon;

    % Now the prediction phase starts
    for hour=1:horizon
        
        % Hour and day type variables that will be used to index the
        % parameters are defined
        Par_hour = times(index+hour,1);
        Par_type = times(index+hour,2);
        
        % A prediction step is carried out for each forecasting algorithm
        Pred.MAPLF = prediction_step_MAPLF(Par, Pred.MAPLF, hour, Par_hour, Par_type,  Prev_hour, observations(index+hour,:)');
        Pred.KF = prediction_step_KF(Par, Pred.KF, hour, Par_hour, Par_type, Prev_hour, observations(index+hour,:)');
        Pred.InvertedSSM = prediction_step_InvertedSSM(Par, Pred.InvertedSSM, hour, Par_hour, Par_type, Prev_hour, observations(index+hour,:)');
        Pred.VAR = prediction_step_VAR(Par, Pred.VAR, hour, Par_hour, Par_type, Prev_hour);

        % The previous hour parameter is updated
        Prev_hour = hour;
        
        % Prediction means are copied in the Predicted_loads variables
        Pred.MAPLF.Predicted_loads(hour,:) = Pred.MAPLF.Mu(:,hour)';
        Pred.KF.Predicted_loads(hour,:) = Pred.KF.Mu(:,hour)';
        Pred.InvertedSSM.Predicted_loads(hour,:) = Pred.InvertedSSM.Mu(:,hour)';
        Pred.VAR.Predicted_loads(hour,:) = Pred.VAR.Mu(:,hour)';

        % Prediction covariances are copied in the Predicted_cov variables
        Pred.MAPLF.Predicted_cov(:,:,hour) = Pred.MAPLF.Sigma(:,:,hour);
        Pred.KF.Predicted_cov(:,:,hour) = Pred.KF.Sigma(:,:,hour);
        Pred.InvertedSSM.Predicted_cov(:,:,hour) = Pred.InvertedSSM.Sigma(:,:,hour);
        Pred.VAR.Predicted_cov(:,:,hour) = Pred.VAR.Sigma(:,:,hour);

        % If one_hour_ahead is true, update the initial parameters Sigma and Mu
        if one_hour_ahead
            Pred.MAPLF.Sigma(:,:,hour) = zeros(number_of_classes, number_of_classes);
            Pred.MAPLF.Mu(:,hour) = loads(index+hour,:)';
            Pred.KF.Sigma(:,:,hour) = zeros(number_of_classes, number_of_classes);
            Pred.KF.Mu(:,hour) = loads(index+hour,:)';
            Pred.InvertedSSM.Sigma(:,:,hour) = zeros(number_of_classes, number_of_classes);
            Pred.InvertedSSM.Mu(:,hour) = loads(index+hour,:)';
            Pred.VAR.Sigma(:,:,hour) = zeros(number_of_classes, number_of_classes);
            Pred.VAR.Mu(:,hour) = loads(index+hour,:)';
        end

    end    

    
    % Now the evaluation metrics are computed
    % First a matrix containing the real observations for the given day is
    % defined
    True_load = loads(index+(1:horizon),:);

    % Then the metrics are updated for each forecasting algorithm
    Pred.MAPLF = prediction_scores(Pred.MAPLF, True_load, qs, day);
    Pred.KF = prediction_scores(Pred.KF, True_load, qs, day);
    Pred.InvertedSSM = prediction_scores(Pred.InvertedSSM, True_load, qs, day);
    Pred.VAR = prediction_scores(Pred.VAR, True_load, qs, day);

    % The true loads are saved
    Pred.True_loads(horizon*(day-1)+(1:horizon),:) = True_load;


    % After having evaluated the predictions, the parameters are trained using the real observations 
    hour=times(index+1,1);
    day_type = times(index+1,2);
    while hour<day_length
        index = index+1;
        hour = times(index,1);
        Par.Transition = train_step(loads(index,:)', loads(index-1,:)', Par.Transition, hour, day_type, true);
        Par.EmissionX = train_step(loads(index,:)', observations(index,:)', Par.EmissionX, hour, day_type, true);
        Par.EmissionY = train_step(observations(index,:)', loads(index,:)', Par.EmissionY, hour, day_type, true);
        Par.InvertedSSM = train_step([loads(index,:)'; observations(index,:)'], loads(index-1,:)', Par.InvertedSSM, hour, day_type, true);
    end

    % The day variable is updated
    day=day+1;
end


% Finally, the prediction scores are divided by the number of prediction
% days
Pred.MAPLF = prediction_correct_scores(Pred.MAPLF, prediction_days);
Pred.KF = prediction_correct_scores(Pred.KF, prediction_days);
Pred.InvertedSSM = prediction_correct_scores(Pred.InvertedSSM, prediction_days);
Pred.VAR = prediction_correct_scores(Pred.VAR, prediction_days);

end