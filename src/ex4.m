
%% Initialization
clear ; close all; clc

%% Setup the parameters
input_layer_size  = 6;
hidden_layer_size = 15;   % 15 hidden units
num_labels = 12;

iteration_start = 1;
iteration_end = 5;
%% =========== Part 1: Loading Data =============.
%

% Load Training Data
fprintf('Loading Data ...\n')

load('auto_rate_1.mat');
nw_para_1 = nw_para_1';
X = [nw_para_1(:,1:2) nw_para_1(:, 4:7)];
y = nw_para_1(:, 3);
m = size(X, 1);
[X, mu, sigma] = featureNormalize(X);

%random permutation
permu = randperm(m);
X(permu,:);
y(permu,:);

training_size = floor(0.5 * m);
X_train = X(1:training_size,:);
X_cv = X(training_size+1:end,:);

y_train = y(1:training_size,:);
y_cv = y(training_size+1:end,:);

%fprintf('Program paused. Press enter to continue.\n');
%pause;


%% ================ Part 2: Initializing Pameters ================

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =================== Part 3: Training NN ===================

fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 100);

parfor i = iteration_start:iteration_end

%try different values of lambda
lambda = i;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%fprintf('Program paused. Press enter to continue.\n');
%pause;

%% ================= Part 4: Implement Predict =================
%compute the training set accuracy.

pred = predict(Theta1, Theta2, X_train);

fprintf('\nTraining Set Accuracy for lamba = %d: %f\n', ...
    i, mean(double(pred == y_train)) * 100);


pred = predict(Theta1, Theta2, X_cv);

fprintf('\nCV Set Accuracy for lamba = %d: %f\n', ...
    i, mean(double(pred == y_cv)) * 100);
%fprintf('\nfinding learning curve, please wait\n');
%learning_curve(X, y, Theta1, Theta2);

end