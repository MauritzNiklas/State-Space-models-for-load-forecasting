%{
Function that runs one single training step for one hour of the day

It requires:
- y: the new observation for the first dimension of the matrix (columns of Theta)
- x: the new observation for the second dimension of the matrix (rows of Theta)
- Par: the parameters structure for the current parameters
- hour: the hour of the day for which the parameters are updated
- day_type: the type of day for which the parameters are updated
- out: a boolean that states whether to compute output parameters Theta and Cov or not

It outputs:
- Par: the structure that contains the updated trained parameters
%}


function [Par] = train_step(y, x, Par, hour, day_type, out)

% If the parameters contain the intercept, 1 is added to the new observation x
if Par.intercept
    x = [1;x];
end

% First, gamma, H, J and K are updated recursively
Par.gamma(hour, day_type) = Par.lambda*Par.gamma(hour, day_type)+1;
Par.H(:,:,hour, day_type) = Par.lambda*Par.H(:,:,hour, day_type) + x*y';
Par.J(:,:,hour, day_type) = Par.lambda*Par.J(:,:,hour, day_type) + x*x';
Par.K(:,:,hour, day_type) = Par.lambda*Par.K(:,:,hour, day_type) + y*y';

% If output parameters are required, matrices Theta and Cov are computed
if out 
    
    Par.Theta(:,:,hour, day_type) = Par.J(:,:,hour, day_type)\Par.H(:,:,hour, day_type);
    Par.Gamma(:,:,hour, day_type) = (Par.K(:,:,hour, day_type) - Par.H(:,:,hour, day_type)'*Par.Theta(:,:,hour, day_type))/Par.gamma(hour, day_type);

end

end