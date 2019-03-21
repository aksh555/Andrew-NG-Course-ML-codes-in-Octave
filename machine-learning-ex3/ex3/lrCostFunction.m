% SC SELF COMMENTS

function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
[r c]=size(theta);
grad = zeros(size(theta));
hypo=sigmoid(X*theta);  % SC computing sigmoid function for all elemnts of matrix
x=-y+1;  
hyp=-hypo+1;
cost=(y'*log(hypo) + x'*log(hyp));  

smtheta=zeros(r-1 ,c);
for i=2:size(theta),
  smtheta(i-1)=theta(i);  % SC using smtheta as regularization does not involve theta0
end
reg=lambda/(2*m)*(sum(smtheta.^2));   % SC regularization term 
J= -1/m*(cost) + reg; 
grad=1/m*(X'*(hypo-y));  % SC finding the normal gradient
for i=2:size(grad),
  grad(i)=grad(i)+(lambda/m*(theta(i,1))); % SC adding reg term for all except theta0
end
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%










% =============================================================

grad = grad(:);

end
