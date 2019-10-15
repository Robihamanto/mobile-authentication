function [classresult, probability] = Testing_SVM_TL(classiferNum, model, testDataAns, testData_expandedHistogram, transformation, binWidth)

%TESTINGMODEL Summary of this function goes here
%   Detailed explanation goes here

%% Dyah Work
transformed_testData_expandedHistogram = testData_expandedHistogram + (repmat(transformation,size(testData_expandedHistogram,1),1));

transformed_testData_expandedHistogram(transformed_testData_expandedHistogram < 0) = 0;
transformed_testData_expandedHistogram(transformed_testData_expandedHistogram > 1) = 1;

%normalization part
normalized_transformed_testData = [];
for transformedDataCount = 1:size(transformed_testData_expandedHistogram, 1)
    normedHist = [];
    binIndex = 1;
    for featureCount = 1:numel(binWidth)
        minBin = binIndex;
        maxBin = binIndex + binWidth(featureCount) -1;
        
        %normalize here
        singleHist = transformed_testData_expandedHistogram(transformedDataCount,minBin:maxBin);
        normedSingleHist = singleHist ./ sum(singleHist(:));
        normedHist = [normedHist normedSingleHist];
        
        binIndex = maxBin + 1;
    end
    normalized_transformed_testData = [normalized_transformed_testData; normedHist];
end


[classresult,probability] = ClassifierToolsTest_forAcc(classiferNum, model, testDataAns, normalized_transformed_testData);

end
