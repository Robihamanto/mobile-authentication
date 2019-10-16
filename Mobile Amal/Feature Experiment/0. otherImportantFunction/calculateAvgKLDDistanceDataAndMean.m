function [avgDistance] = calculateAvgDistanceDataAndMean(data,mean_)

% sizef1 = size(mean_,2);
% sizef2 = size(data,2);
% 
% f1=(1:sizef1)';
% f2=(1:sizef2)';

% totalDistance=[];
totalDistance=0;
for(count=1:size(data,1))
%     [f_1, distance] = emd(f1, f2, mean_, data(count,:), @gdf);
    distance = KLDivergence(mean_,data(count,:));
%     totalDistance=[totalDistance;distance];
    totalDistance=totalDistance+distance;
end
avgDistance = totalDistance/count;
end