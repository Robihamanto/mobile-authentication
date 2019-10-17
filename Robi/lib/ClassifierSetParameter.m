function [ penaltyList, hiddenSizesNum] = ClassifierSetParameter ( classiferNum )

penaltyList = [];
hiddenSizesNum = 0;
%     linearSVM
if classiferNum == 1
    numx = [1:5];
    penaltyList = 10 .^ numx;
%     penaltyList = [1];
% numB = [40:-1:1];
% penaltyList = [1./numB 1];
% penaltyList = [1];
%     CART
elseif classiferNum == 2
     penaltyList = [1];
%      NN
elseif classiferNum == 3
     penaltyList = [1];
     hiddenSizesNum = 25;
elseif classiferNum == 4
    penaltyList = [1];
elseif classiferNum == 5
    penaltyList = [1];
end

end

