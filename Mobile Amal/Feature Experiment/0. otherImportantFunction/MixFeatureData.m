function [ featureData ] = MixFeatureData( data )
%MIXHISTOGRAM Summary of this function goes here
%   Detailed explanation goes here
featureData = [];
for i = 1:size(data,2)
    featureData = [featureData data(i).featureData];
end

end

