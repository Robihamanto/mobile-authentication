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

for dataCount = 1:size(testData, 1)
    
    if ExperimentParameters().isHistogramDataUsed
        expHTest = MixData(testData(dataCount,:));
    else
        expHTest =testData;
    end
    
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
