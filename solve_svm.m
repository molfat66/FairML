function [ b, b0 ] = solve_svm( X, Y, L, B, d, Sig, mu )
%SOLVE_SVM Compute the SVM solution

%% Set default inputs and initialize variables
[n, p] = size(X);

%% Solve the SVM optimization problem
cvx_begin quiet
    variable s(n)
    variable b(p)
    variable b0

    minimize( sum(s)/n + L*sum_square(b) ) 

    Y.*(X*b + b0) >= 1 - s;
    s >= 0

    if (nargin >= 5)
        -d <= B*b <= d
    end
cvx_end

if (nargin == 7)
    [V, D] = eig(Sig);
    pD = D; pD(D < 0) = 0;
    nD = D; nD(D >= 0) = 0;
    pSig = V*pD/V;
    nSig = V*nD/V;
    
    bp = b;
    maxiters = 500;
    while (maxiters > 0)
        cvx_begin quiet
            
            variable s(n)
            variable b(p)
            variable b0
            variable t

            minimize( sum(s)/n + L*sum_square(b) + mu*t ) 

            Y.*(X*b + b0) >= 1 - s;
            s >= 0

            -d <= B*b <= d

            b'*pSig*b + bp'*nSig*bp + 2*bp'*nSig'*(b - bp) <= t
            -b'*nSig*b - bp'*pSig*bp - 2*bp'*pSig'*(b - bp) <= t
        cvx_end  

        if (norm(b - bp)/norm(bp) < 1e-3)
            break;
        else
            bp = b;
            maxiters = maxiters - 1;
        end
    end
end

end

