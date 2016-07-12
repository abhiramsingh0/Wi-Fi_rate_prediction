 function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J.
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively.
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m, 1) X];

Z2 = X * Theta1';
A2 = sigmoid(Z2);
A2 = [ones(m, 1), A2];

Z3 = A2 * Theta2';
A3 = sigmoid(Z3);

ytemp = y;
y = zeros(m ,num_labels);

data_rates = [1,2,5.5,6,9,11,12,18,24,36,48,54];

for i= 1:num_labels
    Index = find(ytemp == data_rates(i));
    y(Index, i) = 1;
end

J = (-1/m) * sum(sum((y .* log(A3) + (1-y) .* log(1-A3)) , 2)) ...
    + (lambda / (2*m)) * (sum(sum(Theta1 .^ 2),2) ...
    + sum(sum(Theta2 .^ 2),2) - sum(Theta1(:,1) .^ 2) ...
    - sum(Theta2(:,1) .^ 2));

delta3 = A3 - y;
delta2 = (delta3 * Theta2(:, 2:end)) .* sigmoidGradient(Z2);

Cdelta2 = delta3' * A2;
Cdelta1 = delta2' * X;

Theta1_grad = (1/m) * Cdelta1;
Theta2_grad = (1/m) * Cdelta2;

temp_grad1 = Theta1_grad(:, 2:end);
temp_grad2 = Theta2_grad(:, 2:end);

reg_grad1 = temp_grad1 + (lambda/m) * Theta1(:, 2:end);
reg_grad2 = temp_grad2 + (lambda/m) * Theta2(:, 2:end);

Theta1_grad = [Theta1_grad(:,1) reg_grad1];
Theta2_grad = [Theta2_grad(:,1) reg_grad2];
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
