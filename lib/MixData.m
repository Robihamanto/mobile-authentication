function [ result ] = MixData( data )

%MIXHISTOGRAM Summary of this function goes here
%   1 for histogram and 0 for feature data

result = [];
if ExperimentParameters().isHistogramDataUsed
    for i = 1:numel(data)
        result = [result data(i).histogram];
    end
else
    for i = 1:numel(data)
        result = [result data(i).featureData];
    end
end