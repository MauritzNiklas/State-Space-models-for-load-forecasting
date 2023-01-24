%{
Function that initialises the required parameters for training

It requires:
- n: the first dimension of the matrix (columns of Theta)
- m: the second dimension of the matrix (rows of Theta)
- day_length: the number of observations contained in a day (e.g. for hourly observations, it's 24)
- day_types: the number of day_types in the times matrix (e.g. 2 for weekdays and weekends)
- lambda: the value for lambda used to train this matrix
- intercept: a boolean that states whether to consider the intercept or not

It outputs:
- Par: the structure that contains the trained parameters
%}


function [Par] = train_initialization(n, m, day_length, day_types, lambda, intercept)

% Initialization of training parameters

% In this part, matrices H, J and Theta are created
% If there is an intercept, the second dimension is m+1
if intercept
    Par.H = zeros(m+1, n, day_length, day_types); 
    Par.J = zeros(m+1, m+1, day_length, day_types); 
    Par.Theta = zeros(m+1, n, day_length, day_types);
else % otherwise, dimension m is kept
    Par.H = zeros(m, n, day_length, day_types); 
    Par.J = zeros(m, m, day_length, day_types); 
    Par.Theta = zeros(m, n, day_length, day_types);
end

% In this part, matrices Gamma, K and Cov and also vector gamma are created
Par.Gamma = zeros(n, n, day_length, day_types); 
Par.K = zeros(n, n, day_length, day_types); 
Par.Cov = zeros(n, n, day_length, day_types);
Par.gamma = zeros(day_length, day_types); 

% Now, the properties of lambda and intercept for this structure are set
Par.lambda = lambda;
Par.intercept = intercept;

end