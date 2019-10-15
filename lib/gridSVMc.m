function [ cost,probability ] = gridSVMc(totalTrunFeatureData, trainDataAns, gridArray)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
  probability = [];
  for index = 1:numel(gridArray)
      cmd = [ '-v 3 -c ',num2str( gridArray(index)), ' -q'];
      fitness(index) = svmtrain(trainDataAns, totalTrunFeatureData, cmd);
  end
  [~,maxindex]=max(fitness);
  cost = gridArray(maxindex);

%   posData = totalTrunFeatureData(trainDataAns == 1,:);
%   negData = totalTrunFeatureData(trainDataAns == 0,:);
%   posDataNum = fix(size(posData,1)/3);
%   negDataNum = fix(size(negData,1)/3);
%   posModdataNum = mod(size(posData,1),3);
%   negModdataNum = mod(size(negData,1),3);
%   posrand = randperm(size(posData,1));
%   negrand = randperm(size(negData,1)); 
%   
%   posIndexTabel = [1 posDataNum;posDataNum+1 posDataNum*2;posDataNum*2+1 (posDataNum*3)+posModdataNum];
%   posSize = [posDataNum;posDataNum;posDataNum+posModdataNum];
%   negIndexTabel = [1 negDataNum;negDataNum+1 negDataNum*2;negDataNum*2+1 (negDataNum*3)+negModdataNum];
%   negSize = [negDataNum;negDataNum;negDataNum+negModdataNum];  
%   libsvmOption = strcat( '-q -s 0 -b 1 -t 0 -c',32 , num2str(cost));
%   Posprobability = [];
%   Negprobability = [];
%   for fordIndex = 1:3
%       trainingData = [];
%       trainingAns = [];
%       testData = [];
%       testAns = [];
%       for takeIndex = 1:3
%           if fordIndex == takeIndex
%               testData = totalTrunFeatureData(posrand(posIndexTabel(takeIndex,1):posIndexTabel(takeIndex,2)),:);
%               testData = [testData;totalTrunFeatureData(negrand(negIndexTabel(takeIndex,1):negIndexTabel(takeIndex,2)),:)];
%               testAns = [testAns;ones(posSize(takeIndex),1);zeros(negSize(takeIndex),1)];
%           else
%               trainingData = [trainingData;totalTrunFeatureData(posrand(posIndexTabel(takeIndex,1):posIndexTabel(takeIndex,2)),:)];
%               trainingData = [trainingData;totalTrunFeatureData(negrand(negIndexTabel(takeIndex,1):negIndexTabel(takeIndex,2)),:)];
%               trainingAns = [trainingAns;ones(posSize(takeIndex),1);zeros(negSize(takeIndex),1)];
%           end
%       end
%        model= svmtrain(trainingAns,trainingData,libsvmOption);
%        [~,~,temp] = svmpredict(testAns,testData, model,'-b 1');
%        Posprobability = [Posprobability;temp(testAns == 1,:)];
%        Negprobability = [Negprobability;temp(testAns == 0,:)];
%   end
%   probability = [Posprobability;Negprobability];
end

