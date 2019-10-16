clear;
addpath('..\0. otherImportantFunction');
coreFolderName = 'NCC';
rootFolderResults = 'Result final NCC Histogram\5rounds100peoples';

%% PARAMETER SETTING
numOfFlick = 5;
% sampleset_path = ['..\..\0506_' num2str(numOfFlick) 'flicks_train_test_allperiod_5round'];
% sampleset_path_Neg = ['..\..\0506_' num2str(numOfFlick) 'flicks_train_test_allperiod_5round'];
sampleset_path = ['..\0918_5flicks_100users_5round'];
sampleset_path_Neg = ['..\0918_5flicks_100users_5round'];
roundSize = 1;
% userInvolved = [1:4 7:12 14:30 35 36 39:57 59:70];
userInvolved = [1:5 7:57 59:102];
FeatureIndex = 1:49;
periodList = [3 4 5 6 7 8];

% 1=linearSVM 2=CART 3=NN 4=Bayse 5=WKNN
classifierList = [1];

%%
for postureCount=1
    if postureCount==1
        posture = 'sit_long';
    elseif postureCount==2
        posture = 'sit_medium';
    elseif postureCount==3
        posture = 'sit_short';
    elseif postureCount==4
        posture = 'stand_long';
    elseif postureCount==5
        posture = 'stand_medium';
    else
        posture = 'stand_short';
    end
    
    folderName = [rootFolderResults '\' posture '\' coreFolderName '\'] ;
    allRound_AvgEer=[];
    
    for roundCount=1:roundSize
        
        %-------------------make folder for FAR, FAR, and EER------------------
        fullFolderName = [folderName coreFolderName num2str(roundCount)];
        folderToMake = ['..\exp_result\' fullFolderName '\eer\eer_1\'];
        mkdir(folderToMake);
        %------------------------------------------------------------------------
        
        %----- if you need folder to detail result, like accuracy each user -----
        %folderToMake2 = ['..\exp_result\' folderN '\userData\4\'];
        %folderToMake3 = ['..\exp_result\' folderN '\userData\5\'];
        %folderToMake4 = ['..\exp_result\' folderN '\userData\6\'];
        %folderToMake5 = ['..\exp_result\' folderN '\userData\7\'];
        %folderToMake6 = ['..\exp_result\' folderN '\userData\8\'];
        %folderToMake7 = ['..\exp_result\' folderN '\userData\3\'];
        %mkdir(folderToMake2);
        %mkdir(folderToMake3);
        %mkdir(folderToMake4);
        %mkdir(folderToMake5);
        %mkdir(folderToMake6);
        %mkdir(folderToMake7);
        %------------------------------------------------------------------------
        
        for classifierIndex = 1:numel(classifierList)
            classifierNum = classifierList(classifierIndex);
            [penaltyList,hiddenSizesNum] = ClassifierSetParameter(classifierNum);
            
            %% LOAD NEGATIVE FOR TRAINING AND TESTING
            load([sampleset_path_Neg '\' posture '\round_' num2str(roundCount) '_train_neg_sampleSet.mat'], 'trainNegHistogram');
            load([sampleset_path_Neg '\' posture '\round_' num2str(roundCount) '_test_neg_sampleSet.mat'], 'testNegHistogram');
            %%
            
            for trainingCount = 1:1
                avgAccuracy=[];
                totalTIME_training = [];
                totalTIME_testing = [];
                
                for testingCount = 1:numel(periodList)
                    correct_rate = [];
                    Average_AR = [];
                    Average_FAR = [];
                    Average_FRR = [];
                    
                    for userCount = 1:numel(userInvolved)
                        
                        userID = userInvolved(userCount);
                        %% LOAD POSITIVE FOR TRAINING AND TESTING
                        % Load positive for training in period training
                        load([sampleset_path '\' posture '\user_' num2str(userID) '_period_' num2str(periodList(trainingCount)) '_round_' num2str(roundCount) '_train_sampleSet.mat'], 'trainHistogram');
                        trainHistogram_old = trainHistogram;
                        % Load positive for training in period testing
                        load([sampleset_path '\' posture '\user_' num2str(userID) '_period_' num2str(periodList(testingCount))  '_round_' num2str(roundCount) '_train_sampleSet.mat'], 'trainHistogram');
                        trainHistogram_new = trainHistogram;
                        trainHistogram_new = getHistogram(trainHistogram_new);
                        % Load positive for testing
                        load([sampleset_path '\' posture '\user_' num2str(userID) '_period_' num2str(periodList(testingCount))  '_round_' num2str(roundCount) '_test_sampleSet.mat'], 'testHistogram');
                        
                        fprintf('Training Period: %d, Testing Period: %d, Training Starting\n', periodList(trainingCount), periodList(testingCount));
                        
                        %% NEGATIVEDATA - REMOVE CURRENT USER DATA FROM trainNegHistogram%
                        trainNegData = trainNegHistogram;
                        trainNegData(userCount,:)=[];
                        trainNegHist = getHistogram(trainNegData);
                        
                        %% NEGATIVEDATA - REMOVE CURRENT USER DATA FROM testNegHistogram
                        testNegData = testNegHistogram;
                        testNegData(userCount,:)=[];
                        testNegHist = getHistogram(testNegData);
                        
                        %% POSITIVEDATA - SET POSITIVE DATA for training and testing
                        trainPosData = trainHistogram_old;
                        testPosData = testHistogram;
                        
                        trainPosHist = getHistogram(trainPosData);
                        testPosHist = getHistogram(testPosData);
                        testDataAns = [ones(size(testPosData,1),1);zeros(size(testNegData,1),1)];
                        
                        %% ======= TRAINING !!! ======
                        trainingTime1 = 0;
                        trainingTime2 = 0;
                        if (periodList(trainingCount)==periodList(testingCount))
                            % when current period is 3, calculate mean neg                                                        
                            t = clock;
                            [meanNeg{userCount},trainNegHist] = getMeanTrainingData(trainNegHist,FeatureIndex);
                            trainingTime1 = etime(clock,t);
                        end
                        
                        %calculate mean positive (new behavior)
                        t2 = clock;
                        [meanData_new,trainHistogram_new] = getMeanTrainingData(trainHistogram_new,FeatureIndex);
                        trainingTime2 = etime(clock,t2);
                        totalTIME_training{userCount,testingCount} = trainingTime1+trainingTime2;
                        
                        %% TESTING
                        t3 = clock;
                        [distanceTestWNewAll,distanceWithNegAll,distanceWithNegMultiAll,classresult,probability,testData] = Nearest_Centroid_Classifier (FeatureIndex,testPosData,testNegData,meanData_new,meanNeg{userCount});
                        totalTIME_testing{userCount,testingCount} = etime(clock,t3);
                        
                        %% DISPLAY
                        
                        trainPosData_expandedHistogram = [];
                        for dataCount = 1:size(trainPosHist,1)
                            %combined all histogram (expanded histogram)
                            trainPosData_expandedHistogram = [trainPosData_expandedHistogram;MixHistogram(trainPosHist(dataCount,:))];
                        end
                        
                        allData = [trainPosData_expandedHistogram;trainNegHist;trainHistogram_new;testData];
                        numberDimensions = 3;
                        % % Use PCA function to reduce dimension of The Raw Feature from 25 dimension to 3 dimension
                        [coef, score,~,~, explained] = pca(allData, 'NumComponents', numberDimensions);
                        
                        trainP = size(trainPosHist,1);%30
                        trainN = size(trainNegHist,1);%99
                        trainNewP = size(trainHistogram_new,1);%30
                        testNewP = size(testPosHist,1);%30
                        testNewN = size(testNegHist,1);%99
                        
                        figure
                        % train old positive
                        scatter3(score(1:trainP,1), ...
                                 score(1:trainP,2), ...
                                 score(1:trainP,3), 'Marker','^', 'MarkerFaceColor','blue')
                        set(gca,'XLim',[-2 2],'YLim',[-2 2],'ZLim',[-2 2])
                        xlabel('X label')
                        ylabel('Y label')
                        zlabel('Z label')
                        grid on
                        hold on
                        %train old negative
                        scatter3(score(trainP+1:trainP+trainN,1), ...
                                 score(trainP+1:trainP+trainN,2), ...
                                 score(trainP+1:trainP+trainN,3), 'Marker','s', 'MarkerFaceColor','red')
                        hold on
                        %train new positive
                        scatter3(score(trainP+trainN+1:trainP+trainN+trainNewP,1), ...
                                 score(trainP+trainN+1:trainP+trainN+trainNewP,2), ...
                                 score(trainP+trainN+1:trainP+trainN+trainNewP,3), 'Marker','^', 'MarkerFaceColor','green')
                        hold on
                        %test new positive
                        scatter3(score(trainP+trainN+trainNewP+1:trainP+trainN+trainNewP+testNewP,1), ...
                                 score(trainP+trainN+trainNewP+1:trainP+trainN+trainNewP+testNewP,2), ...
                                 score(trainP+trainN+trainNewP+1:trainP+trainN+trainNewP+testNewP,3), 'Marker','^', 'MarkerFaceColor','green')
                        hold on
                        %test new negative
                        scatter3(score(trainP+trainN+trainNewP+testNewP+1:trainP+trainN+trainNewP+testNewP+testNewN,1), ...
                                 score(trainP+trainN+trainNewP+testNewP+1:trainP+trainN+trainNewP+testNewP+testNewN,2), ...
                                 score(trainP+trainN+trainNewP+testNewP+1:trainP+trainN+trainNewP+testNewP+testNewN,3), 'Marker','s', 'MarkerFaceColor','yellow')
                        saveas(gcf,sprintf('Figure user %d round %d.png' ,userTestCount,round));
                        
                        %% calculate FAR, FRR AND ACCURACY
                        thresholdtable = repmat(0:0.01:1,numel(testDataAns),1);
                        datatable = repmat(probability(:,1),1,101);
                        testingResult = datatable>thresholdtable;
                        
                        correctRate = [];
                        FAR = [];
                        FRR = [];
                        for i = 1:size(thresholdtable,2)
                            [ c, fa, Fr ] = MobileCorrectRate_FAR_FRR(testingResult(:,i),testDataAns,size(testPosData, 1));   %?????FAR?FRR
                            correctRate = [correctRate c];
                            FAR = [FAR fa];
                            FRR = [FRR Fr];
                        end
                        
                        Average_AR = [Average_AR;correctRate];
                        Average_FAR = [Average_FAR;FAR];
                        Average_FRR = [Average_FRR;FRR];
                        
                        %----- if you need folder to detail result, like accuracy each user -----
                        %if (period(trainingCount)~=period(testingCount))
                        %filename = ['..\exp_result\' folderN '\userData\' num2str(period(testingCount)) '\user' num2str(userID)];
                        %excelName = [filename '.xlsx'];
                        %xlswrite(excelName,classresult,'accuracy','A1');
                        %xlswrite(excelName,distanceTestWNewAll,'accuracy','B1');
                        %xlswrite(excelName,distanceWithNegAll,'accuracy','C1');
                        %end
                        %------------------------------------------------------------------------
                        
                        classTestPos = ones(size(testPosData,1),1);
                        classTestNeg = zeros(size(testNegData,1),1);
                        
                        CP = classperf([classTestPos;classTestNeg], classresult);
                        
                        correct_rate = [correct_rate;CP.CorrectRate];
                    end
                    
                    %save accuracy to file
                    filename = ['..\exp_result\' fullFolderName '\TLaccu_Train' num2str(periodList(trainingCount)) '_test' num2str(periodList(testingCount))];
                    excelName = [filename '.xlsx'];
                    xlswrite(excelName,correct_rate,'CorrectRate','A1');
                    xlswrite(excelName,mean(correct_rate),'CorrectRate','D1');
                    avgAccuracy = [avgAccuracy mean(correct_rate)];
                    
                    
                    % save FAR and FRR to file
                    filename = ['..\exp_result\' fullFolderName '\eer\eer_1\TLEER_Train' num2str(periodList(trainingCount)) '_test' num2str(periodList(testingCount))];
                    A2 = {'ACR'};
                    A3 = {'FAR'};
                    A4 = {'FRR'};
                    excelName = [filename '.xlsx'];
                    xlswrite(excelName,A2,'ACR','A1');
                    xlswrite(excelName,A3,'FAR','A1');
                    xlswrite(excelName,A4,'FRR','A1');
                    xlswrite(excelName,Average_AR,'ACR','B1');
                    xlswrite(excelName,Average_FAR,'FAR','B1');
                    xlswrite(excelName,Average_FRR,'FRR','B1');
                end
                
                % save testing time
                totalTIME_testing = cell2mat(totalTIME_testing);
                meanTestingTime = mean(totalTIME_testing);
                totalTIME_testing= [totalTIME_testing; meanTestingTime];
                filename =['..\exp_result\' fullFolderName '\testingTime'];
                excelName =[filename '.xlsx'];
                xlswrite(excelName,totalTIME_testing,'testingTime','A1');
                
                % save training time
                totalTIME_training = cell2mat(totalTIME_training);
                meanTrainingTime = mean(totalTIME_training);
                totalTIME_training = [totalTIME_training; meanTrainingTime];
                filename = ['..\exp_result\' fullFolderName '\trainingTime'];
                excelName = [filename '.xlsx'];
                xlswrite(excelName,totalTIME_training,'trainingTime','A1');
                % compute EER and save it
                [allPeriodAvgEer]=ComputeInformation(fullFolderName);
                allRound_AvgEer = [allRound_AvgEer;allPeriodAvgEer];
                
                % save accuracy to file
                filename = ['..\exp_result\' fullFolderName '\averageAccu'];
                excelName = [filename '.xlsx'];
                xlswrite(excelName,avgAccuracy,'AvgAccu','A1');
            end
        end
        
    end
    
    %save average EER for all ROUND
    filename = ['..\exp_result\' folderName '\averageEER'];
    excelName = [filename '.xlsx'];
    xlswrite(excelName,allRound_AvgEer,'AvgEER','A1');
    
end