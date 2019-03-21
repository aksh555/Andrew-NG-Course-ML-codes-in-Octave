function [J,grad] = comp(X, y, theta)

m = length(y);
J = 0;
[r c]=size(theta);
grad = zeros(size(theta));
hypo = X*theta;
error = (hypo-y)'*(hypo-y);
J = 1/(2*m)*error;
grad=1/m*(X'*(X*theta-y));
end
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1);
initial_theta = zeros(size(X, 2), 1);

[J,grad]=comp(X, y, theta);
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, J, exit_flag] = ...
	fminunc(@(t)(comp(t, X, y, lambda)), initial_theta, options);
fprintf("%f",theta);
