function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix



%SC here what we are actually doing is first assigning initial_theta to zero which is the initial assumtion
% before finding optimal values. then y is the matrix containing the actual digit which img represnts for each training eg.
% so we need to implement 10 binary logistic regressions to find optimal theta values for each classifier 0 through 9
% for each regression we create a new matrix yval which has element value =1 if y=that particular digit
% classifier else 0 thus converting it to standard binary kinda problem. 
% then fmincg finds the optimal theta values for that particular classifier label which is stored as the corresponding
% row in all_theta which is the matrix of order 10 x n+1 where each row has the optimal theta values for each label.




X = [ones(m, 1) X];
for i=1:num_labels,
  initial_theta=zeros(n+1,1);
  yval=double(i==y); % checks if i==y and makes yval 1 if true else 0.(binary)
  options = optimset('GradObj', 'on', 'MaxIter', 400);
  [theta] = ...
        fmincg (@(t)(lrCostFunction(t, X, yval, lambda)), ...
                 initial_theta, options);
  all_theta(i,:)=theta';
end
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%












% =========================================================================


end
