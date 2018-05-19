function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

values = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
numValues = numel(values);
numCom = numValues^2;
errors = zeros(numValues^2,3);
i = 1;

for indexC = 1:numValues
  for indexSigma = 1:numValues    
    C = values(indexC);
    sigma =values(indexSigma);
    fprintf('\n\nCombination %d of %d; C = %d, sigma = %d\n######################################\n',i,numCom,C,sigma);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    errors(i,:) = [C,sigma,error];
    i = i + 1;
  end
end

[minValue,rownum]=min(errors);
minRowIndex = rownum(3);
C = errors(minRowIndex,1);
sigma = errors(minRowIndex,2);

% =========================================================================

end
