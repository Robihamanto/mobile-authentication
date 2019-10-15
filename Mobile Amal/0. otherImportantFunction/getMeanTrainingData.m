function [meanNeg,meanPos,meanNegMulti] = getMeanTrainingData(trainPosData,trainNegData,featureSet)

trainPosData = trainPosData(:,featureSet);
trainNegData = trainNegData(:,featureSet);
trainData = [trainPosData;trainNegData];

trainData_expandedHistogram = [];
for dataCount = 1:size(trainData,1)
    %combined all histogram (expanded histogram)
    trainData_expandedHistogram = [trainData_expandedHistogram;MixHistogram(trainData(dataCount,:))];
end

meanNeg = mean(trainData_expandedHistogram(size(trainPosData,1)+1:size(trainData,1),:));
meanPos = mean(trainData_expandedHistogram(1:size(trainPosData,1),:));
meanNegMulti=0;
% [~,meanNegMulti]=kmeans(trainData_expandedHistogram(size(trainPosData,1)+1:size(trainData,1),:),3);