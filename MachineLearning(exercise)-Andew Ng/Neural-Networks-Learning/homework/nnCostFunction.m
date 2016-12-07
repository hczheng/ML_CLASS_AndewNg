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
    X = [ones(m, 1) X];
    temp=sigmoid(X*Theta1');
    temp=[ones(m, 1) temp];
    temp=sigmoid(temp*Theta2');
%      [~,temp]=max(temp');
%       temp=temp';
%   将y变形
   newy=zeros(m,num_labels);
   for i = 1:m
       for j=1:num_labels
          if(y(i)==j) newy(i,j)=1;
          end
       end  
   end
   J=-newy*log(temp')-(1-newy)*log(1-temp');
%  求矩阵的迹
    J=trace(J)/m;
% tmp1=ones(size(Theta1'));
% tmp1(1,:)=0; 
% tmp2=ones(size(Theta2'));
% tmp2(1,:)=0; 
 Tx1=Theta1';
 Tx2=Theta2';
z1 = zeros(1,hidden_layer_size);
z2 = zeros(1,num_labels);
Tx1(1,:)=z1;%Theta1正则化（第一行置0）
Tx2(1,:)=z2;%Theta2正则化（第一行置0）
Txx1=Tx1.^2;
Txx2=Tx2.^2;
J=J+lambda*(sum(Txx1(:))+sum(Txx2(:)))/(2*m);

% -------------------------------------------------------------
% Backpropagation:1、2、3、4、5
% feedforward（a1,a2,a3）
 a1=X;
 a2=sigmoid(a1*Theta1');
 a2=[ones(m, 1) a2];
 a3=sigmoid(a2*Theta2');
 % δ(3)
 tk3=a3-newy;
 % δ(2)，delta 2 = delta 2(2:end).
 tk2=tk3*Theta2.*(a2.*(1-a2));
 tk2=tk2(:,2:hidden_layer_size+1);
 % ? (l) = ?(l)+δ(l+1)(a(l))T
 d1=tk2'*a1;
 d2=tk3'*a2;
 Theta1_grad=d1/m+Tx1'*lambda/m;%Theta1_grad正则化（第一行置0），Tx1
 Theta2_grad=d2/m+Tx2'*lambda/m;%Theta1_grad正则化（第一行置0），Tx2
% =========================================================================
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
