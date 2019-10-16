function [distanceTestWNewAll,distanceWithNegAll,distanceWithNegMultiAll,classresult,probabilityPos] = Nearest_Centroid_Classifier(featureSet,testPosData,testNegData,meanData_new,meanNeg)

%TESTINGMODEL Summary of this function goes here
%   Detailed explanation goes here

testPosData = testPosData(:,featureSet);
testNegData = testNegData(:,featureSet);

testData = [testPosData; testNegData];

distanceTestWNewAll=[];
distanceWithNegAll=[];
distanceWithNegMultiAll=[];
distanceTestWNew=[];
distanceWithNeg=[];
classresult=[];
probabilityPos=[];
% testPosData_expandedFeatureData=[];
% 
% for dataCount = 1:size(testPosData, 1)
%     testPosData_expandedFeatureData = [testPosData_expandedFeatureData; MixFeatureData(testPosData(dataCount,:))];
% end
% 
% testPosData_expandedFeatureData = normc(testPosData_expandedFeatureData);
% 
% testNegData_expandedFeatureData=[];
% 
% for dataCount = 1:size(testNegData, 1)
%     testNegData_expandedFeatureData = [testNegData_expandedFeatureData; MixFeatureData(testNegData(dataCount,:))];
% end
% 
% testNegData_expandedFeatureData = normc(testNegData_expandedFeatureData);
% 
% testData_expandedFeatureData = [testPosData_expandedFeatureData;testNegData_expandedFeatureData];

for dataCount = 1:size(testData, 1)
    %% euclidian distance
    distanceTestWNew = calculateAvgEuDistanceDataAndMean(testData(dataCount,:), meanData_new);
    distanceWithNeg = calculateAvgEuDistanceDataAndMean(testData(dataCount,:), meanNeg);
    
    if (distanceTestWNew<=distanceWithNeg)
        classresult = [classresult; 1];
    else
        classresult = [classresult; 0];
    end
    probP = distanceWithNeg/(distanceWithNeg+distanceTestWNew);
    
    probabilityPos = [probabilityPos;probP];
    %% Resume
    distanceTestWNewAll=[distanceTestWNewAll;distanceTestWNew];
    distanceWithNegAll=[distanceWithNegAll;distanceWithNeg];
    distance = [distanceTestWNewAll distanceWithNegAll];
%     distanceWithNegMultiAll=[distanceWithNegMultiAll;distanceWithNegMulti];
end

end
