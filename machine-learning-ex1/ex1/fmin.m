data = load('ex1data1.txt');
m=size(y);
y = data(:, 2);
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1);
initial_theta = zeros(size(X, 2), 1);

[J,grad]=comp(theta,X,y);
fprintf("%f\n",J);
options = optimset('GradObj', 'on', 'MaxIter', 1500);
[theta, J, exit_flag] = ...
	fminunc(@(t)(comp(t, X, y, lambda)), initial_theta, options);
fprintf("%f\n",theta);