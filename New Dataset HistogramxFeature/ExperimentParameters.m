classdef ExperimentParameters < handle
    %EXPERIMENTPARAMETERS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties 
        isHistogramDataUsed = false; % true for histogram false for feature data
        postures = {'sit_short', 'sit_medium', 'sit_long', 'stand_short', 'stand_medium', 'stand_long'};
        userInvolved = [1:5 7:57 59:102];
    end
    
    methods
        
    end
end

