function [model] = Training(featureSet,trainPosData,trainNegData,classiferNum,penaltyList,hiddenSizesNum,numOfFlick)

trainPosData = trainPosData(:,featureSet);
trainNegData = trainNegData(:,featureSet);
trainData = [trainPosData;trainNegData];

trainData_expandedFeatureData = [];
for dataCount = 1:size(trainData,1)
    trainData_expandedFeatureData = [trainData_expandedFeatureData;MixData(trainData(dataCount,:))];
end

if ExperimentParameters().isHistogramDataUsed == false
    trainData_expandedFeatureData = normc(trainData_expandedFeatureData);
end

trainData_expandedFeatureData = normc(trainData_expandedFeatureData);

if ExperimentParameters().isHistogramDataUsed
    trainDataAns = [ones(size(trainPosData,1),1);zeros(size(trainNegData,1),1)];
else 
    trainDataAns = [ones(numOfFlick * size(trainPosData,1),1);zeros(numOfFlick * size(trainNegData,1),1)];
end
    

[model] = ClassifierToolsTraining(classiferNum,penaltyList,trainData_expandedFeatureData,trainDataAns,hiddenSizesNum);

end

