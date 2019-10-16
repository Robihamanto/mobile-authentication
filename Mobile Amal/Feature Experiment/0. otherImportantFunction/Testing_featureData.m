function [classResult, probability,testDataAns] = Testing_FeatureData(featureSet,testPosData,testNegData,model,classiferNum,numOfFlick)
   
%TESTINGMODEL Summary of this function goes here
%   Detailed explanation goes here

testPosData = testPosData(:,featureSet);
testNegData = testNegData(:,featureSet);
testData = [testPosData;testNegData];

testDataAns = [ones(size(testPosData,1),1);zeros(size(testNegData,1),1)];
% 
% testPosData_expandedFeatureData = [];
% for dataCount = 1:size(testPosData, 1)
%     testPosData_expandedFeatureData = [testPosData_expandedFeatureData; MixFeatureData(testPosData(dataCount,:))];
% end
% 
% testPosData_expandedFeatureData = normc(testPosData_expandedFeatureData);
% 
% testNegData_expandedFeatureData = [];
% for dataCount = 1:size(testNegData, 1)
%     testNegData_expandedFeatureData = [testNegData_expandedFeatureData; MixFeatureData(testNegData(dataCount,:))];
% end
% 
% testNegData_expandedFeatureData = normc(testNegData_expandedFeatureData);
% 
% testData_expandedFeatureData = [testPosData_expandedFeatureData;testNegData_expandedFeatureData];

[classResult, probability] = ClassifierToolsTest_forAcc(classiferNum, model, testDataAns, testData);

end



