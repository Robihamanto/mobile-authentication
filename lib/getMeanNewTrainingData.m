function [meanPos] = getMeanNewTrainingData(featureSet,newTrainingData)

trainData = newTrainingData(:,featureSet);

trainData_expandedHistogram = [];
for dataCount = 1:size(trainData,1)
    %combined all histogram (expanded histogram)
    trainData_expandedHistogram = [trainData_expandedHistogram;MixHistogram(trainData(dataCount,:))];
end

meanPos = mean(trainData_expandedHistogram(:,:));