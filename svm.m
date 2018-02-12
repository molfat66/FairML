function [ b, b0, L ] = svm( X, Y, L_vec, K )
%SVM Compute linear support vector machine

%% Set default inputs and initialize variables
if (nargin == 2)
    L_vec = logspace(-5,1,10);
    K = 5;
elseif (nargin == 3)
    K = 5;
end

[n, p] = size(X);

%% Generate K-Folds
ndK = floor(n/K);

fold_inds = randperm(n);
fold_inds = fold_inds(1:K*ndK);
fold_inds = reshape(fold_inds, ndK, K);

%% Perform K-Fold cross validation
cv_err = zeros(1, length(L_vec));

for ind = 1:K
    train_inds = fold_inds(:, setdiff(1:K, ind));
    train_inds = train_inds(:);
    test_inds = fold_inds(:, ind);
    
    pind_test = (Y(test_inds) >= 0);
    nind_test = (Y(test_inds) <= 0);

    for ind_j = 1:length(L_vec)
        [b, b0] = solve_svm(X(train_inds,:), Y(train_inds), L_vec(ind_j));

        vn = (X(test_inds,:)*b + b0 >= 0);
        cv_err(ind_j) = cv_err(ind_j) + sum(1-vn(pind_test))/sum(pind_test) + sum(vn(nind_test))/sum(nind_test);
    end
end

%% Compute the final SVM
[~, mind] = min(cv_err);
L = L_vec(mind);
[b, b0] = solve_svm(X, Y, L);        

end