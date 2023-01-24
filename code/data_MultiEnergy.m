%{
Function that returns the required data from the Tempe University dataset

It outputs:
- loads: the matrix that contains the loads
- observations: the matrix that contains the additional observations (e.g. the temperature)
- times: the matrix that contains the time variables (e.g. hour and day type)
- day_length: the number of observations contained in a day (e.g. for hourly observations, it's 24)
- day_types: the number of day_types in the times matrix (e.g. 2 for weekdays and weekends)
- dates: datetime variable related to the dataset
- loads_min_max: structure will is useful to change normalisation
%}


function [loads, observations, times, day_length, day_types] = data_MultiEnergy()

% Load the data
load("Tempe_data.mat")


% Define loads and observations
loads = [electric cooling heating];
observations = methereologic_table{:,9};

% Normalize the observations
[~,number_of_obs_variables] = size(observations);
for i=1:number_of_obs_variables
    observations(:,i) = (observations(:,i)-min(observations(:,i)))/(max(observations(:,i))-min(observations(:,i)));
end

% Create the times matrix
num_days = length(electric)/24;
times = repmat([(1:24)', ones(24,1)],num_days,1);

% Define day lenght and types of day
day_length = 24;
day_types = 1;

end