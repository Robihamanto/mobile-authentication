function [ histogram ] = MixHistogram( data )
%MIXHISTOGRAM Summary of this function goes here
%   Detailed explanation goes here
histogram = [];
for i = 1:numel(data)
    histogram = [histogram data(i).histogram];
end

end

