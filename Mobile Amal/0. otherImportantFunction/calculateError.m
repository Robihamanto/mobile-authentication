function [typeI, typeII] = calculateError(testProbability,thresholdList)
thresholdtable = repmat(thresholdList,size(testProbability,1),1);
    datatable = repmat(testProbability(:,1),1,101);
    testingResult = datatable>thresholdtable;
    
    typeI = [];
    typeII = [];
    %calculate type I error and type II error for particular threshold
    for i = 1:numel(thresholdList)
        %calculate type II error for positive data
        typeIIthisThreshold = numel(find(testingResult(1:size(testProbability,1)/2, i) == 0));
        typeII = [typeII typeIIthisThreshold];
      
        %calculate type I error for negative data
        typeIthisThreshold = numel(find(testingResult(size(testProbability,1)/2 + 1:size(testProbability,1), i) == 1));
        typeI = [typeI typeIthisThreshold];
    end
end