function [classResult, probability,testDataAns] = Testing_FeatureData(featureSet,testPosData,testNegData,model,classiferNum,numOfFlick)
   
%TESTINGMODEL Summary of this function goes here
%   Detailed explanation goes here

testPosData = testPosData(:,featureSet);
testNegData = testNegData(:,featureSet);
testData = [testPosData;testNegData];

testDataAns = [ones(numOfFlick*size(testPosData,1),1);zeros(numOfFlick*size(testNegData,1),1)];

testData_expandedFeatureData = [];
for dataCount = 1:size(testData, 1)
    testData_expandedFeatureData = [testData_expandedFeatureData; MixFeatureData(testData(dataCount,:))];
end
% testData_expandedFeatureData(isnan(testData_expandedFeatureData)) = 0;

testData_expandedFeatureData = normc(testData_expandedFeatureData);

[classResult, probability] = ClassifierToolsTest_forAcc(classiferNum, model, testDataAns, testData_expandedFeatureData);

end



