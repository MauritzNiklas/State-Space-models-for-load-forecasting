%{
Function that returns the required data from the ISO New England dataset

It requires:
- type: type of dataset to return
- normalised: boolean variable, if true a 0-1 normalization is done
- no_weekends: boolean variable, if true weekends and workdays are the same

It outputs:
- loads: the matrix that contains the loads
- observations: the matrix that contains the additional observations (e.g. the temperature)
- times: the matrix that contains the time variables (e.g. hour and day type)
- day_length: the number of observations contained in a day (e.g. for hourly observations, it's 24)
- day_types: the number of day_types in the times matrix (e.g. 2 for weekdays and weekends)
- dates: datetime variable related to the dataset
- loads_min_max: structure will is useful to change normalisation
%}


function [loads, observations, times, day_length, day_types, dates, loads_min_max] = data_MultiRegions(type, normalised, no_weekends)

% Load the ISO New England dataset
load("ISONE.mat")

% Define loads and observations from the ISONE dataset. If:
% - type is aggregated, return the aggregated loads and temperatures
% - type is agg_l_sub_o, return the subregional loads with the
% aggregated temperatures
% - type is subregional, return the subregional loads and temperatures
if type == "aggregate"
    loads = Total_loads;
    observations = Total_obs;
elseif type == "agg_l_sub_o"
    loads = Diff_loads(:,[1,2,3,4,5,7,8,9]);
    observations = Total_obs;
elseif type == "subregional"
    loads = Diff_loads(:,[1,2,3,4,5,7,8,9]);
    observations = Diff_obs(:,[1,2,3,4,5,6,7,8,9,10,13,14,15,16,17,18]);
else
    disp('Error: no correct type of dataset')
end

% Extract the times matrix
times = [times(:,1) times(:,4)+1];

% Extract the number of variables
[~,number_of_obs_variables] = size(observations);
[~,number_of_classes] = size(loads);

% Define the loads_min_max structure, which will be only useful to change
% normalisation
loads_min_max = zeros(number_of_classes+1,2);

% If normalised is true, make a 0-1 normalization on the data
if normalised

    % Normalise the observations matrix
    for i=1:number_of_obs_variables
        observations(:,i) = (observations(:,i)-min(observations(:,i)))/(max(observations(:,i))-min(observations(:,i)));
    end
    
    % Normalise the load matrix
    for i=1:number_of_classes
        loads_min_max(i,1) = min(loads(:,i));
        loads_min_max(i,2) = max(loads(:,i));
        loads(:,i) = (loads(:,i)-min(loads(:,i)))/(max(loads(:,i))-min(loads(:,i)));
    end

    % Add the values to the loads_min_max variable
    loads_min_max(number_of_classes+1,1) = min(Total_loads);
    loads_min_max(number_of_classes+1,2) = max(Total_loads);
end

% Define the day information parameters
day_length = 24;
day_types = 2;

% If no_weekends is true, only one type of day will be present (i.e.
% weekends and work days are considered the same)
if no_weekends
    times(:,2) = 1;
    day_types = 1;
end



end