function [meanNeg] = getMeanTrainingData(trainData,featureSet)

trainData = trainData(:,featureSet);
trainData_expandedHistogram = [];

if ExperimentParameters().isHistogramDataUsed
    for dataCount = 1:size(trainData,1)
        %combined all histogram (expanded histogram)
        trainData_expandedHistogram = [trainData_expandedHistogram;MixData(trainData(dataCount,:))];
    end
    meanNeg = mean(trainData_expandedHistogram(1:size(trainData,1),:));
else 
    meanNeg = mean(trainData(1:size(trainData,1),:));
end
end