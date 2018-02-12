function [ alph, k ] = solve_ksvm( X, Y, Kmat, L, B, d, Sig, mu )
%SOLVE_KSVM Compute the kernel SVM solution

%% Set default inputs and initialize variables
[n, p] = size(X);

%% Solve the SVM optimization problem
cvx_begin quiet
    variable alph(n)

    minimize( (alph.*Y)'*Kmat*(alph.*Y)/n - sum(alph)/n  ) 

    sum(alph.*Y) == 0
    0 <= alph <= L

    if (nargin >= 6)
        -d <= B*alph <= d
    end
cvx_end

if (nargin == 8)
    Sig = Sig/max(max(abs(Sig)));
    [V, D] = eig(Sig);
    pD = D; pD(D <= 0) = 0;
    nD = D; nD(D >= 0) = 0;
    pSig = V*pD*V';
    nSig = V*nD*V';
    
    bp = alph.*Y;
    maxiters = 500;
    while (maxiters > 0)
        cvx_begin quiet
            variable alph(n)
            variable t

            minimize( (alph.*Y)'*Kmat*(alph.*Y)/n - sum(alph)/n + mu*t ) 
            
            sum(alph.*Y) == 0
            0 <= alph <= L

            -d <= B*alph <= d

            (alph.*Y)'*pSig*(alph.*Y) + bp'*nSig*bp + 2*bp'*nSig'*(alph.*Y - bp) <= t
            -(alph.*Y)'*nSig*(alph.*Y) - bp'*pSig*bp - 2*bp'*pSig'*(alph.*Y - bp) <= t            
        cvx_end  
        
        if (norm(alph.*Y - bp)/norm(bp) < 1e-3)
            break;
        else
            bp = alph.*Y;
            maxiters = maxiters - 1;
        end
    end
end

%% Identify an unconstrained coefficient
k = find(0 < alph & alph < L);

end

