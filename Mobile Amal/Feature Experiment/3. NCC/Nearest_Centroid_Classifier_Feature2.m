function [distanceTestWNewAll,distanceWithNegAll,distanceWithNegMultiAll,classresult,probabilityPos] = Nearest_Centroid_Classifier(featureSet,testPosData,testNegData,meanData_new,meanNeg)

%TESTINGMODEL Summary of this function goes here
%   Detailed explanation goes here

% testPosData = testPosData(:,featureSet);
% testNegData = testNegData(:,featureSet);

testData = [testPosData; testNegData];

distanceTestWNewAll=[];
distanceWithNegAll=[];
distanceWithNegMultiAll=[];
distanceTestWNew=[];
distanceWithNeg=[];
classresult=[];
probabilityPos=[];
% testData_expandedFeatureData=[];
% 
% for dataCount = 1:size(testData, 1)
%     testData_expandedFeatureData = [testData_expandedFeatureData; MixFeatureData(testData(dataCount,:))];
% end
% 
% testData_expandedFeatureData = normc(testData_expandedFeatureData);

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
