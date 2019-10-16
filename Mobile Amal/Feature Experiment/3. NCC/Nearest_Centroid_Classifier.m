function [distanceTestWNewAll,distanceWithNegAll,distanceWithNegMultiAll,classresult,probabilityPos,expHTestAll] = Nearest_Centroid_Classifier(featureSet,testPosData,testNegData,meanData_new,meanNeg)

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
expHTestAll = [];

for dataCount = 1:size(testData, 1)
    expHTest = MixHistogram(testData(dataCount,:));
    expHTestAll = [expHTestAll;expHTest];
    %% euclidian distance
    distanceTestWNew = calculateAvgEuDistanceDataAndMean(expHTest, meanData_new);
    distanceWithNeg = calculateAvgEuDistanceDataAndMean(expHTest,meanNeg);
    
    if (distanceTestWNew<=distanceWithNeg)
        classresult = [classresult; 1];
    else
        classresult = [classresult; 0];
    end
    probP = distanceWithNeg/(distanceWithNeg+distanceTestWNew);
    
    probabilityPos = [probabilityPos;probP];
    %% Resume
%     distanceTestWNewAll=[distanceTestWNewAll;distanceTestWNew];
%     distanceWithNegAll=[distanceWithNegAll;distanceWithNeg];
%     distanceWithNegMultiAll=[distanceWithNegMultiAll;distanceWithNegMulti];
end

end
