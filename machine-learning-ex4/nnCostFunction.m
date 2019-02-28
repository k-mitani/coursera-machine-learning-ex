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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2. 
%

# JÇãÅÇﬂÇÈ
Xbiased = [ones(size(X, 1), 1) X]; # (5000, 401)
Z2 = Xbiased * Theta1';
A2 = sigmoid(Z2); # (5000, 25)
A2biased = [ones(size(A2, 1), 1) A2]; # (5000, 26)
Z3 = A2biased * Theta2'; # (5000, 10)
A3 = sigmoid(Z3); # (5000, 10)

# yÇOneHotâªÇ∑ÇÈ
yOneHot = zeros(m, num_labels);
for i = 1:m
  yOneHot(i, y(i)) = 1;
endfor

# J (without Regularization term)ÇãÅÇﬂÇÈ
JwithoutRt = 1 / m * sum(sum(-yOneHot .* log(A3) - (1 - yOneHot) .* log(1 - A3)));

# Regularization termÇãÅÇﬂÇÈÅB
Theta1WithoutBias = Theta1(:, 2:end);
Theta2WithoutBias = Theta2(:, 2:end);
rt = lambda / (2 * m) * (sum(sum(Theta1WithoutBias .* Theta1WithoutBias)) + sum(sum(Theta2WithoutBias .* Theta2WithoutBias)));

J = JwithoutRt + rt;
% -------------------------------------------------------------

# delta3ÇãÅÇﬂÇÈ (5000, 10)
Delta3 = A3 - yOneHot;
# delta2ÇãÅÇﬂÇÈ (5000, 26)
Delta2 = Delta3 * Theta2 .* sigmoidGradient([ones(size(Z2, 1), 1) Z2]);
# biasÇèúÇ≠ (5000, 25)
Delta2 = Delta2(:, 2:end);
# gradÇãÅÇﬂÇÈ (10, 26)
Theta2_grad = 1 / m .* (Delta3' * A2biased) + lambda / m .* [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
# gradÇãÅÇﬂÇÈ (25, 401)
Theta1_grad = 1 / m .* (Delta2' * Xbiased) + lambda / m .* [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
