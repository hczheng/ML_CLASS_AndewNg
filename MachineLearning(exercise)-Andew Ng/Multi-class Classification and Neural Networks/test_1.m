% % input:
% theta = [-2; -1; 1; 2];
% X = [ones(3,1) magic(3)];
% y = [1; 0; 1] >= 0.5;       % creates a logical array
% lambda = 3;
% [J grad] = lrCostFunction(theta, X, y, lambda)
% output:
% J = 7.6832
% grad =
%    0.31722
%    -0.12768
%    2.64812
%    4.23787

%input:
% X = [magic(3) ; sin(1:3); cos(1:3)];
% y = [1; 2; 2; 1; 3];
% num_labels = 3;
% lambda = 0.1;
% [all_theta] = oneVsAll(X, y, num_labels, lambda)
%output:
% all_theta =
%   -0.559478   0.619220  -0.550361  -0.093502
%   -5.472920  -0.471565   1.261046   0.634767
%    0.068368  -0.375582  -1.652262  -1.410138

% input:
all_theta = [1 -6 3; -2 4 -3];
X = [1 7; 4 5; 7 8; 1 4];
predictOneVsAll(all_theta, X)
%output:
% ans =
%    1
%    2
%    2
%    1

