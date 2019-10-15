function [avgDistance] = calculateAvgDistanceDataAndMean(data,mean_)

totalDistance=0;
for(count=1:size(data,1))
    distance = sqrt(sum((data(count,:) - mean_) .^ 2));
    totalDistance=totalDistance+distance;
end
avgDistance = totalDistance/count;
end