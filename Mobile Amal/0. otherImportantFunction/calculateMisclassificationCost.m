function [minimumCost, threshold] = calculateMisclassificationCost(typeI, typeII, costRatio, thresholdList)
    MCAllThreshold = typeI * costRatio + typeII;
    [minimumCost,index] = min(MCAllThreshold);
    
    threshold = thresholdList(1,index);
end
