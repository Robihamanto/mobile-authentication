function [ KLD ] = KLDivergence(P,Q)
%KLDIVERGENCE Summary of this function goes here
%   Detailed explanation goes here

if(numel(P)~=numel(Q))
    error('All inputs must have same dimension.');
end
KLD = 0;


for intervalcount = 1:numel(P)
    if(P(intervalcount) == 0)
        P(intervalcount) = P(intervalcount) + eps;
    end
    if(Q(intervalcount) == 0)
        Q(intervalcount) = Q(intervalcount) + eps;
    end
end
%P = P ./ sum(P);
%Q = Q ./ sum(Q);
LOG2_P = log2(P);
LOG2_Q = log2(Q);
PdQ = LOG2_P - LOG2_Q;
KLD = sum(P .* PdQ);

QdP = LOG2_Q - LOG2_P;
KLD = KLD + sum(Q .* QdP);


%{
for intervalindex = 1:numel(Q)
    if Q(intervalindex) == 0
        KLD = KLD + 0;
    elseif P(intervalindex) == 0 
        P(intervalindex) = P(intervalindex) + eps;
        KLD = KLD + ( Q(intervalindex) * (log2( Q(intervalindex)) - log2(P(intervalindex)))  );
    else
        KLD = KLD + ( Q(intervalindex) * (log2( Q(intervalindex)) - log2(P(intervalindex)))  );
    end    
end
%}

end

