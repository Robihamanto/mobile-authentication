function [featureDataOutput] = getFeatureData( data )
%MIXHISTOGRAM Summary of this function goes here
%   Detailed explanation goes here
for i = 1:size(data, 1)
    for j = 1:size(data,2)
        featureDataOutput(i,j).featureData = data(i,j).featureData;  
    end
end
end

