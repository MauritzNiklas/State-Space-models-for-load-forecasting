clear
close all

% Load the data from the ISO New England dataset
% In this case, the subregional loads and the aggregated temperatures will
% be returned
type = "agg_l_sub_o";
[loads, observations, times, day_length, day_types, dates, loads_min_max] = data_MultiRegions(type, true, false);

%{
To change the dataset to the Tempe dataset, the following functions may be used:
[loads, observations, times, day_length, day_types] = data_MultiEnergy();

Keep in mind that the code is not optimised for this dataset
%}


% Instantiate the parameters that are necessary for the training of the
% parameters
training_days = 365*2; % 2 years
starting_day = datetime(2004,1,1);
start_index = find(dates==starting_day);
lambdas = [1 1 1 1];
intercepts = [1 1 1 1];

% Procede to train the parameters
[Par, index] = train(loads, observations, times, training_days, day_length, day_types, lambdas, intercepts, start_index);


% Instantiate the parameters that are necessary for the prediction phase
prediction_days = 365*3; % 3 years
horizon = day_length;
starting_hour = 1;
one_hour_ahead = false;

% Procede to the daily prediction + training phase
[Pred, Par, index] = prediction(Par, loads, observations, times, prediction_days, horizon, starting_hour, one_hour_ahead, day_length, index);


% Print the results
disp("MAPE of model with p(X_t|Y_t) and p(X_t|X_{t-1}) is "+mean(Pred.MAPLF.MAPE,'all'))
disp("MAPE of model with p(Y_t|X_t) and p(X_t|X_{t-1}) is "+mean(Pred.KF.MAPE,'all'))
disp("MAPE of model with p(X_t,Y_t|X_{t-1}) is "+mean(Pred.InvertedSSM.MAPE,'all'))
disp("MAPE of VAR model on X_t is "+mean(Pred.VAR.MAPE,'all'))

disp("RMSE of model with p(X_t|Y_t) and p(X_t|X_{t-1}) is "+sqrt(mean(Pred.MAPLF.MSE,'all')))
disp("RMSE of model with p(Y_t|X_t) and p(X_t|X_{t-1}) is "+sqrt(mean(Pred.KF.MSE,'all')))
disp("RMSE of model with p(X_t,Y_t|X_{t-1}) is "+sqrt(mean(Pred.InvertedSSM.MSE,'all')))
disp("RMSE of VAR model on X_t is "+sqrt(mean(Pred.VAR.MSE,'all')))

disp("LogScore of model with p(X_t|Y_t) and p(X_t|X_{t-1}) is "+mean(Pred.MAPLF.LogScore))
disp("LogScore of model with p(Y_t|X_t) and p(X_t|X_{t-1}) is "+mean(Pred.KF.LogScore))
disp("LogScore of model with p(X_t,Y_t|X_{t-1}) is "+mean(Pred.InvertedSSM.LogScore))
disp("LogScore of VAR model on X_t is "+mean(Pred.VAR.LogScore))

disp("Pinball loss of model with p(X_t|Y_t) and p(X_t|X_{t-1}) is "+mean(Pred.MAPLF.PinballLoss(:,:,:),'all'))
disp("Pinball loss of model with p(Y_t|X_t) and p(X_t|X_{t-1}) is "+mean(Pred.KF.PinballLoss(:,:,1),'all'))
disp("Pinball loss of model with p(X_t,Y_t|X_{t-1}) is "+mean(Pred.InvertedSSM.PinballLoss(:,:,1),'all'))
disp("Pinball loss of VAR model on X_t is "+mean(Pred.VAR.PinballLoss(:,:,1),'all'))

%% Plot predictions

colors = [227, 99, 93; 90, 169, 230; 255, 228, 94; 96, 211, 148; 248, 249, 250]/255;

class = 6;
start_day = 30;
final_day = 37;

intv = 24*start_day+1:24*final_day;

figure('Renderer', 'painters', 'Position', [10 10 900 450])
colororder(colors)
tcl = tiledlayout(1,1);
nexttile(tcl)
plot(dates(index+intv+starting_hour),[Pred.MAPLF.Total_predictions(intv, class), ...
    Pred.KF.Total_predictions(intv, class), ...
    Pred.InvertedSSM.Total_predictions(intv, class), ...
    Pred.VAR.Total_predictions(intv, class)],'LineWidth',1.5)
hold on
plot(dates(index+intv+starting_hour),Pred.True_loads(intv, class),':k','LineWidth',2)
hold off

xlabel('Day')
ylabel('NE Massachusetts normalized demand')
xline(dates(index+1+24*(start_day-1:final_day)), '--k','LineWidth',1.3)
xlim([dates(index+starting_hour+24*start_day+1),dates(index+starting_hour-1+24*final_day+1)])
hL = legend('MAPLF', 'KF', 'Inv SSM', 'VAR', 'Real load','TextColor','k','FontSize',12);
hL.Layout.Tile = 'East';
grid on;
