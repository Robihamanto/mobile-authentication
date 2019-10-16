function [meanNeg,trainData_expandedHistogram] = getMeanTrainingData(trainData,featureSet)


trainData = trainData(:,featureSet);

trainData_expandedHistogram = [];
for dataCount = 1:size(trainData,1)
    %combined all histogram (expanded histogram)
    trainData_expandedHistogram = [trainData_expandedHistogram;MixHistogram(trainData(dataCount,:))];
end

meanNeg = mean(trainData_expandedHistogram(1:size(trainData,1),:));

end