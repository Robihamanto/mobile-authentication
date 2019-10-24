function [ model,probability ] = ClassifierToolsTraining(classiferNum, penaltyNum, totalTrunFeatureData, trainDataAns, hiddenSizesNum)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
cost = 0;
probability = [];
if classiferNum == 1
    [cost] = gridSVMc(totalTrunFeatureData, trainDataAns, penaltyNum);
    libsvmOption = strcat( '-q -s 0 -b 1 -t 2 -c',32 , num2str(cost));
    model.model = svmtrain(trainDataAns,totalTrunFeatureData,libsvmOption);
elseif classiferNum == 2
    cost = [0 penaltyNum ; 1 0];
    flag = num2str(trainDataAns);
    model = classregtree(totalTrunFeatureData,flag,'cost',cost);
elseif classiferNum == 3
    tempModel = feedforwardnet(hiddenSizesNum);
    tempModel.trainParam.showWindow = false;
    tempModel = train(tempModel,totalTrunFeatureData',trainDataAns');
    model.model = tempModel;
    num_NN= sim(tempModel,totalTrunFeatureData');
    model.threshold = linspace(min(num_NN),max(num_NN),20);
elseif classiferNum == 4
    model = NaiveBayes.fit(totalTrunFeatureData,trainDataAns,'dist','kernel');
elseif classiferNum == 5
     threadholdNumIndex = 75;

       delegateDist = prctile(totalTrunFeatureData,threadholdNumIndex);       
        
        distList = [];
        for frac = 0.02:0.02:1
            distList = [distList;delegateDist*(frac)];
        end
        
        for frac = 1.05:0.05:1.25
            distList = [distList;delegateDist*(frac)];
        end
        
        for frac = 1.5:0.25:15
            distList = [distList;delegateDist*(frac)];
        end
        
        for frac = 20:5:200
            distList = [distList;delegateDist*(frac)];
        end
        
        for frac = 200:10:10000
            distList = [distList;delegateDist*(frac)];
        end
        model.distList = distList(:,1:ceil(size(distList,2)/2));

end


end

