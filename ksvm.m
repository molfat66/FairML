function [ alph, k, L ] = ksvm( X, Y, Kmat, L_vec, K )
%SVM Compute kernel support vector machine

%% Set default inputs and initialize variables
if (nargin == 3)
    L_vec = logspace(-5,1,10);
    K = 5;
elseif (nargin == 4)
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
    
    Kmat_train = Kmat(train_inds,:); Kmat_train = Kmat_train(:,train_inds);
    Kmat_test = Kmat(test_inds,:); Kmat_test = Kmat_test(:,train_inds);

    for ind_j = 1:length(L_vec)
        [alph, k] = solve_ksvm(X(train_inds,:), Y(train_inds), Kmat_train, L_vec(ind_j));

        [score, b0] = kpredict(X(train_inds,:), Y(train_inds), alph, k, Kmat_train, Kmat_test);
        vn = (score + b0 >= 0);
        cv_err(ind_j) = cv_err(ind_j) + sum(1-vn(pind_test))/sum(pind_test) + sum(vn(nind_test))/sum(nind_test);
    end
end

%% Compute the final SVM
[~, mind] = min(cv_err);
L = L_vec(mind);
[alph, k] = solve_ksvm(X, Y, Kmat, L);     

end