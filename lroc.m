function [ roc, sroc ] = lroc( X, Y, Z, b )
%lroc Compute ROC curve for main and side response of a linear classifier

%% Set default inputs and initialize variables
pind = (Y >= 0);
nind = (Y < 0);

spind = (Z >= 0);
snind = (Z < 0);

[n, p] = size(X);

L = min(X*b)-1e-5;
U = max(X*b)+1e-5;

%% Compute the ROC

if (norm(b) > 0)
    roc = zeros(1000, 2);
    sroc = zeros(1000, 2);
    b0_vec = linspace(-U,-L,1000);

    for ind = 1:1000
        b0 = b0_vec(ind);
        vn = (X*b + b0 >= 0);

        roc(ind,1) = sum(vn(nind))/sum(nind);
        roc(ind,2) = sum(vn(pind))/sum(pind);    

        sroc(ind,1) = sum(vn(snind))/sum(snind);
        sroc(ind,2) = sum(vn(spind))/sum(spind);    
    end
else
    roc = [0 0; 1 1];
    sroc = roc;
end

end

