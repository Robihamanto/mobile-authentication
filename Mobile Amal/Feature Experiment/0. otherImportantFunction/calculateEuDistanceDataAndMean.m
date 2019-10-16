function [totalDistance] = calculateEuDistanceDataAndMean(data,mean_)

totalDistance=[];
for(count=1:size(data,1))
    distance = sqrt(sum((data(count,:) - mean_) .^ 2));
    totalDistance=[totalDistance;distance];
end
end