function [classResult, probability] = Testing(featureSet,testPosData,testNegData,model,classiferNum,numOfFlick)

testPosData = testPosData(:,featureSet);
testNegData = testNegData(:,featureSet);
testData = [testPosData;testNegData];


if ExperimentParameters().isHistogramDataUsed
    testDataAns = [ones(size(testPosData,1),1);zeros(size(testNegData,1),1)];
else 
    testDataAns = [ones(numOfFlick * size(testPosData,1),1);zeros(numOfFlick * size(testNegData,1),1)];
end

testData_expandedFeatureData = [];
for dataCount = 1:size(testData, 1)
    testData_expandedFeatureData = [testData_expandedFeatureData; MixData(testData(dataCount,:))];
end

if ExperimentParameters().isHistogramDataUsed == false
    testData_expandedFeatureData = normc(testData_expandedFeatureData);
end

testData_expandedFeatureData = normc(testData_expandedFeatureData);

[classResult, probability] = ClassifierToolsTest(classiferNum, model, testDataAns, testData_expandedFeatureData);

end



