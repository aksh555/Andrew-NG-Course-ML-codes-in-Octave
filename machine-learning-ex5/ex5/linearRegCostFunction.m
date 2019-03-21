function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

hypo = X*theta;
error = (hypo-y)'*(hypo-y);
cost = 1/(2*m)*error;

[r c]=size(theta);
smtheta=zeros(r-1 ,c);
for i=2:size(theta),
  smtheta(i-1)=theta(i);
end
reg=lambda/(2*m)*(sum(smtheta.^2));
J= cost + reg;

grad=1/m*(X'*(hypo-y));
for i=2:size(grad),
  grad(i)=grad(i)+(lambda/m*(theta(i,1)));
end


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
