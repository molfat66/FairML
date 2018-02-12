%% Choose CVX solver
cvx_solver Mosek
            
%% Load wine quality data
data = csvread('Wine_Quality_Data.csv', 2);
X = data(:, 1:11);
Y = 2*data(:, 12) - 1;
Z = 2*data(:, 13) - 1;

%% Split data into training and validation sets
[n, p] = size(X);

inds = randperm(n);
tinds = inds(1:floor(0.8*n));
vinds = inds(floor(0.8*n)+1:end);

Xt = X(tinds,:);
Yt = Y(tinds,:);
Zt = Z(tinds,:);

Xv = X(vinds,:);
Yv = Y(vinds,:);
Zv = Z(vinds,:);

%% Choose fairness level
d = 0;
mu = 1e2;

%% Compute (regular) SVM and its ROC curve
disp('Linear SVM')
[b, b0, L] = svm(Xt, Yt);
[roc, sroc] = lroc(Xv, Yv, Zv, b);
del = max(abs(sroc(:,1)-sroc(:,2)))
auc = trapz(roc(:,1), roc(:,2))

subplot(131);
plot(roc(:,1), roc(:,2), 'LineStyle', '-', 'Color', [0    0.4470    0.7410]);
hold on;
plot(sroc(:,1), sroc(:,2), 'LineStyle', '-.', 'Color', [0.8500    0.3250    0.0980]);
plot(linspace(0,1,10), linspace(0,1,10), 'LineStyle', '--', 'Color', [0.5 0.5 0.5]);
hold off;
axis square;

%% Compute (average) fair SVM and its ROC curve
disp('Linear SVM with Linear Fairness Constraint')
spind = (Zt >= 0);
snind = (Zt < 0);
aveX = sum(Xt(spind,:))/sum(spind) - sum(Xt(snind,:))/sum(snind);
[b, b0] = solve_svm( Xt, Yt, L, aveX/norm(aveX), d );
[roc, sroc] = lroc(Xv, Yv, Zv, b);
del = max(abs(sroc(:,1)-sroc(:,2)))
auc = trapz(roc(:,1), roc(:,2))

subplot(132);
plot(roc(:,1), roc(:,2), 'LineStyle', '-', 'Color', [0    0.4470    0.7410]);
hold on;
plot(sroc(:,1), sroc(:,2), 'LineStyle', '--', 'Color', [0.8500    0.3250    0.0980]);
plot(linspace(0,1,10), linspace(0,1,10), 'LineStyle', ':', 'Color', [0 0 0]);
hold off;
axis square;

%% Compute (dc algorithm) fair SVM and its ROC curve
disp('Linear SVM from Spectral Algorithm')
pSigma = cov(Xt(spind,:));
nSigma = cov(Xt(snind,:));
aveX = sum(Xt(spind,:))/sum(spind) - sum(Xt(snind,:))/sum(snind);
[b, b0] = solve_svm( Xt, Yt, L, aveX/norm(aveX), d, pSigma-nSigma, mu );
[roc, sroc] = lroc(Xv, Yv, Zv, b);
del = max(abs(sroc(:,1)-sroc(:,2)))
auc = trapz(roc(:,1), roc(:,2))

subplot(133);
plot(roc(:,1), roc(:,2), 'LineStyle', '-', 'Color', [0    0.4470    0.7410]);
hold on;
plot(sroc(:,1), sroc(:,2), 'LineStyle', '--', 'Color', [0.8500    0.3250    0.0980]);
plot(linspace(0,1,10), linspace(0,1,10), 'LineStyle', ':', 'Color', [0 0 0]);
hold off;
axis square;
