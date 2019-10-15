function [ classResult, probability] = ClassifierToolsTest( classiferNum, testUseModel, testingAns, totalTrunFeatureData )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
if classiferNum == 1    
    [classResult,xxx,probability] = svmpredict(testingAns,totalTrunFeatureData, testUseModel.model,'-b 1');
elseif classiferNum == 2
    result = eval(testUseModel,totalTrunFeatureData);
    testingResult = str2double(result);
elseif classiferNum == 3
%     testingResult = round(abs(sim(testUseModel,totalTrunFeatureData')))';
    testModel = testUseModel.model;
    threshold = testUseModel.threshold;
    threshold = sort(threshold,2,'descend');
    num_NN= sim(testModel,totalTrunFeatureData');
    testingResult = [];
    for dataindex=1:size(num_NN,2)
        tempResult = num_NN(dataindex)>threshold;
        testingResult = [testingResult;tempResult];
    end
elseif classiferNum == 4
          peopleCompute =  posterior(testUseModel,totalTrunFeatureData,'HandleMissing','on');
     peopleCompute = peopleCompute(:,2);
     threshold = [1:-0.02:0];
     testingResult = [];
     for dataindex=1:size(peopleCompute,1)
        tempResult = peopleCompute(dataindex)>threshold;
        testingResult = [testingResult;tempResult];
    end
elseif classiferNum == 5
    distList = testUseModel.distList;
    testingResult = [];
    for dataindex=1:size(totalTrunFeatureData,1)
        testData = totalTrunFeatureData(dataindex,1:ceil(size(totalTrunFeatureData,2)/2));
        tempResult = zeros(size(distList,1),size(distList,2));
        testData = repmat(testData,size(distList,1),1);
        tempResult1 = distList > testData;
        tempResult2 = sum(tempResult1,2);
        tempResult3(tempResult2>=1) = 1;
        tempResult3(tempResult2<1) = 0;
        testingResult = [testingResult;tempResult3];
    end
end
end

