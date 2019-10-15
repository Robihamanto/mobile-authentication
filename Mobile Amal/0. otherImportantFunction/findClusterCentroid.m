function [TrainDataCentroid] = findClusterCentroid(KofKmean,distance,cluster,expandedHistogram)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
    TrainDataCentroid = [];

    for kmeanindex = 1:KofKmean
        %Get alldata's distance on certain index (1 to KofKmean)
        alldistance = distance(:,kmeanindex);
        
		%clusterDistance save all distance that belongs to recent K (in the loop or kmeanindex)
        clusterDistance = alldistance(cluster(:,1) == kmeanindex,:);

		%(?)combine histogram from 1-10
        %expandedHistogramData=histogram 2483
		
		%save the value of histogram of certain cluster that have the same index with kmeanindex
        clusterOfExpTrainData = expandedHistogram(cluster(:,1) == kmeanindex,:);

		%if a cluster doesnt have member because of emptyaction --> singleton
		%insert all data
        if size(clusterDistance,1) == 0
            clusterDistance = alldistance;
            clusterOfExpTrainData = expandedHistogram;
        end
		
		%find the closest data to the centroid
        [~,minindex] = min(clusterDistance);
		
		%save it
        minClusterData = clusterOfExpTrainData(minindex,:);

        TrainDataCentroid = [TrainDataCentroid;minClusterData];
     
    end

end

