function [model] = Training_featureData(featureSet,trainPosData,trainNegData,classiferNum,penaltyList,hiddenSizesNum,numOfFlick)

%take feature data needed (certain featureSet)
trainPosData = trainPosData(:,featureSet);
trainNegData = trainNegData(:,featureSet);
trainData = [trainPosData;trainNegData];

% trainPosData_expandedFeatureData = [];
% for dataCount = 1:size(trainPosData,1)
% 	%combined all histogram (expanded histogram)
%     trainPosData_expandedFeatureData = [trainPosData_expandedFeatureData;MixFeatureData(trainPosData(dataCount,:))];
% end
% trainPosData_expandedFeatureData = normc(trainPosData_expandedFeatureData);

% trainNegData_expandedFeatureData = [];
% for dataCount = 1:size(trainNegData,1)
% 	%combined all histogram (expanded histogram)
%     trainNegData_expandedFeatureData = [trainNegData_expandedFeatureData;MixFeatureData(trainNegData(dataCount,:))];
% end
% 
% trainNegData_expandedFeatureData = normc(trainNegData_expandedFeatureData);

% trainData_expandedFeatureData = [trainPosData_expandedFeatureData;trainNegData_expandedFeatureData];

trainDataAns = [ones(size(trainPosData,1),1);zeros(size(trainNegData,1),1)];

[model,probability] = ClassifierToolsTraining(classiferNum,penaltyList,trainData,trainDataAns,hiddenSizesNum);

end

