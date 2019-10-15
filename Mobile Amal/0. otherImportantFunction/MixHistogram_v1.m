function [ histogram ] = MixHistogram( data )
%MIXHISTOGRAM Summary of this function goes here
%   Detailed explanation goes here
histogram = [];
%% data pake struct
for i = 1:numel(data)
    histogram = [histogram data(i).histogram];
end
%% data pake cell
% histogram = cell2mat(data);
end

