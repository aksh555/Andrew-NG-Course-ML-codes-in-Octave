function [J,grad] = comp(theta,X,y)

m = length(y);
J = 0;
[r c]=size(theta);
grad = zeros(size(theta));
hypo = X*theta;
error = (hypo-y)'*(hypo-y);
J = 1/(2*m)*error;
grad=1/m*(X'*(X*theta-y));
end

