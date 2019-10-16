function [meanNeg] = getMeanTrainingData(trainData,featureSet)


% trainData = trainData(:,featureSet);
% 
% trainData_expandedHistogram = [];
% for dataCount = 1:size(trainData,1)
%     %combined all histogram (expanded histogram)
%     trainData_expandedHistogram = [trainData_expandedHistogram;MixFeatureData(trainData(dataCount,:))];
% end
% 
% trainData_expandedHistogram = normc(trainData_expandedHistogram);

meanNeg = mean(trainData);

end