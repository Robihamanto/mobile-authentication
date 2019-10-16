function [histogramOutput] = getHistogram( data )
%MIXHISTOGRAM Summary of this function goes here
%   Detailed explanation goes here
for i = 1:size(data, 1)
    for j = 1:size(data,2)
        histogramOutput(i,j).histogram = data(i,j).histogram;  
    end
end
end

