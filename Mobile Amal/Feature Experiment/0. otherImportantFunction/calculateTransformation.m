function [transformation,meanData_new] = calculateTransformation(featureSet, trainHistogram_old, trainHistogram_new)

trainHistogram_old = trainHistogram_old(:,featureSet);
trainHistogram_new = trainHistogram_new(:,featureSet);

expandedPosHistogram_old = [];
for dataCount = 1:size(trainHistogram_old,1)
        expandedPosHistogram_old = [expandedPosHistogram_old;MixHistogram(trainHistogram_old(dataCount,:))];
end

expandedPosHistogram_new = [];
for dataCount = 1:size(trainHistogram_new,1)
        expandedPosHistogram_new = [expandedPosHistogram_new;MixHistogram(trainHistogram_new(dataCount,:))];
end

meanData_old = mean(expandedPosHistogram_old);
meanData_new = mean(expandedPosHistogram_new);

transformation = meanData_old - meanData_new;

end