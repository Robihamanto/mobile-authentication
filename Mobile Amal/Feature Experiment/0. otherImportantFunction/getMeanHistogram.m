function [meanData] = getMeanHistogram(data,featureSet)

newData = data(:,featureSet);

expandedHistogram = [];
for dataCount = 1:size(newData,1)
    %combined all histogram (expanded histogram)
    expandedHistogram = [expandedHistogram;MixHistogram(newData(dataCount,:))];
end

expandedHistogram = normc(expandedHistogram);

meanData = mean(expandedHistogram);