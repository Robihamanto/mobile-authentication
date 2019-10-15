function [model] = Training_featureData(featureSet,trainPosData,trainNegData,classiferNum,penaltyList,hiddenSizesNum,numOfFlick)

%take feature data needed (certain featureSet)
trainPosData = trainPosData(:,featureSet);
trainNegData = trainNegData(:,featureSet);
trainData = [trainPosData;trainNegData];

trainData_expandedFeatureData = [];
for dataCount = 1:size(trainData,1)
	%combined all histogram (expanded histogram)
    trainData_expandedFeatureData = [trainData_expandedFeatureData;MixFeatureData(trainData(dataCount,:))];
end
% trainData_expandedFeatureData(isnan(trainData_expandedFeatureData)) = 0;

trainData_expandedFeatureData = normc(trainData_expandedFeatureData);

trainDataAns = [ones(numOfFlick*size(trainPosData,1),1);zeros(numOfFlick*size(trainNegData,1),1)];

[model,probability] = ClassifierToolsTraining(classiferNum,penaltyList,trainData_expandedFeatureData,trainDataAns,hiddenSizesNum);

end

