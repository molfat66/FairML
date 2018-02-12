function [ roc, sroc ] = kroc( Xt, Yt, Kt, Kv, Y, Z, alph, k )
%kroc Compute ROC curve for main and side response of a kernel classifier

%% Set default inputs and initialize variables
pind = (Y >= 0);
nind = (Y < 0);

spind = (Z >= 0);
snind = (Z < 0);

[score, b0] = kpredict(Xt, Yt, alph, k, Kt, Kv);

L = min(score)-1e-5;
U = max(score)+1e-5;

%% Compute the ROC

    roc = zeros(1000, 2);
    sroc = zeros(1000, 2);
    b0_vec = linspace(-U,-L,1000);

    for ind = 1:1000
        b0 = b0_vec(ind);
        vn = (score + b0 >= 0);

        roc(ind,1) = sum(vn(nind))/sum(nind);
        roc(ind,2) = sum(vn(pind))/sum(pind);    

        sroc(ind,1) = sum(vn(snind))/sum(snind);
        sroc(ind,2) = sum(vn(spind))/sum(spind);    
    end


end

