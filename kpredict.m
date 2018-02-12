function [ score, b0 ] = kpredict( X, Y, alph, k, Kmat_train, Kmat_test )
%KPREDICT Predict category of new points with kernel SVM

b0 = mean(Y(k) - Kmat_train(k, :)*(alph.*Y));
score = Kmat_test*(alph.*Y);

end

