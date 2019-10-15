function [model] = Training_pureSVM(featureSet,trainPosData,trainNegData,classiferNum,penaltyList,hiddenSizesNum)

%take feature data needed (certain featureSet)
trainPosData = trainPosData(:,featureSet);
trainNegData = trainNegData(:,featureSet);
trainData = [trainPosData;trainNegData];

trainData_expandedHistogram = [];
for dataCount = 1:size(trainData,1)
	%combined all histogram (expanded histogram)
    trainData_expandedHistogram = [trainData_expandedHistogram;MixHistogram(trainData(dataCount,:))];
end
trainDataAns = [ones(size(trainPosData,1),1);zeros(size(trainNegData,1),1)];

%pca
% [~,score,~,~,~,~] = pca(trainData_expandedHistogram);
% trainData_expandedHistogram = score(:,1:2);

[model,probability] = ClassifierToolsTraining(classiferNum,penaltyList,trainData_expandedHistogram,trainDataAns,hiddenSizesNum);

end

