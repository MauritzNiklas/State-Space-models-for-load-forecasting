%{
Function that runs the initial recursive training procedure

It requires:
- loads: the matrix that contains the loads
- observations: the matrix that contains the additional observations (e.g. the temperature)
- times: the matrix that contains the time variables (e.g. hour and day type)
- training_days: the number of days to train the model's parameters
- day_length: the number of observations contained in a day (e.g. for hourly observations, it's 24)
- day_types: the number of day_types in the times matrix (e.g. 2 for weekdays and weekends)
- lambdas: a vector containing the value of lambda for each parameter matrix
- intercepts: a boolean vector which determines whether to add the intercept or not for each parameter matrix
- start_index: the row index of the loads matrix at the start of the training

It outputs:
- Par: structure that contains the trained parameters
- index: the row index of the loads matrix at the end of the training

This function uses a while loop instead of a for loop to update the parameters because for daylight saving time, there are certain days which may contain one hour more or one hour less
%}


function [Par, index] = train(loads, observations, times, training_days, day_length, day_types, lambdas,  intercepts, start_index)

% Extract the number of variables in the load matrix (e.g. the number of regions)
[~, number_of_classes] = size(loads);
% Extract the number of variables in the observations matrix (e.g. the types of temperatures)
[~, number_of_obs_variables] = size(observations);

% First, the training parameters structure is initialised
% There are 4 parameter structures to train: 
% - the parameters of the transition distribution
% - the parameters of the emission distribution for MAPLF (EmissionX)
% - the parameters of the emission distribution for Kalman Filter (EmissionY)
% - the parameters of the whole distribution for the Inverted SSM model
Par.Transition = train_initialization(number_of_classes, number_of_classes, day_length, day_types, lambdas(1), intercepts(1));
Par.EmissionX = train_initialization(number_of_classes, number_of_obs_variables, day_length, day_types, lambdas(2), intercepts(2));
Par.EmissionY = train_initialization(number_of_obs_variables, number_of_classes, day_length, day_types, lambdas(3), intercepts(3));
Par.InvertedSSM = train_initialization(number_of_classes + number_of_obs_variables, number_of_classes, day_length, day_types, lambdas(4), intercepts(4));

% The training process starts now
% The day, index, hour and day_type variables are defined
day = 1;
index = start_index;
hour = times(index,1); % the hour is extracted from the first training index
day_type = times(index,2); % the type of day is extracted of the first training index

% In the first hour only the emission parameters are updated (since they do not depend on the previous observation)
Par.EmissionX = train_step(loads(index,:)', observations(index,:)', Par.EmissionX, hour, day_type, false);
Par.EmissionY = train_step(observations(index,:)', loads(index,:)', Par.EmissionY, hour, day_type, false);

% Then, for all the training days apart from the last week, the parameters are trained
% In this case the training occurs without output parameters (last parameter is put to false) since it's more computationally efficient
while day<training_days-6
    hour=0;
    % for each day, the parameters are trained until the last hour of the day (i.e. day_length) is reached
    while hour<day_length 
        index = index+1; % the current index is updated
        hour = times(index,1); % consequently the hour of the day is extracted from the index

        % Now all parameters are updated with the current observations
        Par.Transition = train_step(loads(index,:)', loads(index-1,:)', Par.Transition, hour, day_type, false);
        Par.EmissionX = train_step(loads(index,:)', observations(index,:)', Par.EmissionX, hour, day_type, false);
        Par.EmissionY = train_step(observations(index,:)', loads(index,:)', Par.EmissionY, hour, day_type, false);
        Par.InvertedSSM = train_step([loads(index,:)'; observations(index,:)'], loads(index-1,:)', Par.InvertedSSM, hour, day_type, false);
    end
    day = day+1; % after having trained the parameters for the current day, the day parameter is updated
    day_type = times(index+1,2); % the day_type for the next day is extracted
end


% Finally, all parameters for the last week's observations are trained
% In this case, the output parameters are computed (last parameters is put to true), since these parameters will be required for prediction
while day<training_days+1
    hour=0;
    while hour<day_length
        index = index+1;
        hour = times(index,1);
        Par.Transition = train_step(loads(index,:)', loads(index-1,:)', Par.Transition, hour, day_type, true);
        Par.EmissionX = train_step(loads(index,:)', observations(index,:)', Par.EmissionX, hour, day_type, true);
        Par.EmissionY = train_step(observations(index,:)', loads(index,:)', Par.EmissionY, hour, day_type, true);
        Par.InvertedSSM = train_step([loads(index,:)'; observations(index,:)'], loads(index-1,:)', Par.InvertedSSM, hour, day_type, true);
    end
    day = day+1;
    day_type = times(index+1,2);
end

end