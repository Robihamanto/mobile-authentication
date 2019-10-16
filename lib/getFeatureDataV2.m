function [expandedFeatureData] = getFeatureData( data )
%MIXHISTOGRAM Summary of this function goes here
%   Detailed explanation goes here


for i = 1:size(data, 1)
    for j = 1:size(data,2)
        if isempty(data(i,j).featureData)
            data(i,j).featureData = [0;0;0;0;0];
        end
        featureDataOutput(i,j).featureData = data(i,j).featureData;  
    end
end

expandedFeatureData = [];
for dataCount = 1:size(featureDataOutput,1)
	%combined all histogram (expanded histogram)
    expandedFeatureData = [expandedFeatureData;MixFeatureData(featureDataOutput(dataCount,:))];
end

end

