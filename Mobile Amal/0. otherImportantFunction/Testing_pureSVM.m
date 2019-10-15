function [classResult,probability] = Testing_pureSVM(featureSet,testPosData,testNegData,model,classiferNum)
   
%TESTINGMODEL Summary of this function goes here
%   Detailed explanation goes here

testPosData = testPosData(:,featureSet);
testNegData = testNegData(:,featureSet);
testData = [testPosData;testNegData];

testDataAns = [ones(size(testPosData,1),1);zeros(size(testNegData,1),1)];

testData_expandedHistogram = [];
for dataCount = 1:size(testData, 1)
    testData_expandedHistogram = [testData_expandedHistogram; MixHistogram(testData(dataCount,:))];
end

% [~,score,~,~,~,~] = pca(testData_expandedHistogram);
% testData_expandedHistogram = score(:,1:2);

[classResult,probability] = ClassifierToolsTest_forAcc(classiferNum, model, testDataAns, testData_expandedHistogram);

end



