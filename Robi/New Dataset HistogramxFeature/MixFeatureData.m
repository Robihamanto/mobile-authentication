function [ featureData ] = MixFeatureData( data )
%MIXHISTOGRAM Summary of this function goes here
%   1 for histogram and 2 for feature data

featureData = [];
for i = 1:numel(data)
    featureData = [featureData data(i).featureData];
end

end

