function [X_norm ,mu,sigma] = TrialForNorm(X)


X_norm=X;
[r c]=size(X);
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
%[r c]=size(X);
mu(1,1)=((sum(X))(1,1))/r;
sigma(1,1)=std(X(:,1));
for i=1:r,
X_norm(i,1)=(X_norm(i,1)-mu(1,1))/sigma(1,1);
end
%fprintf("mu %f\n",mu(1,1));
%fprintf("si %f\n",sigma(1,1));
%fprintf("%f \n",X_norm);
end