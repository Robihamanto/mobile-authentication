clear;
addpath('..\0. otherImportantFunction');
coreFolderName = 'NCC';
rootFolderResults = 'Result final NCC Feature 2\5rounds100peoples';

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
                        trainHistogram_new = getFeatureData(trainHistogram_new);
                        trainHistogram_new = trainHistogram_new(:,FeatureIndex);

                        expandedHistogram = [];
                        for dataCount = 1:size(trainHistogram_new,1)
                            %combined all histogram (expanded histogram)
                            expandedHistogram = [expandedHistogram;MixFeatureData(trainHistogram_new(dataCount,:))];
                        end
                        
                        trainHistogram_new = normc(expandedHistogram);
                        
                        % Load positive for testing
                        load([sampleset_path '\' posture '\user_' num2str(userID) '_period_' num2str(periodList(testingCount))  '_round_' num2str(roundCount) '_test_sampleSet.mat'], 'testHistogram');
                        
                        fprintf('Training Period: %d, Testing Period: %d, Training Starting\n', periodList(trainingCount), periodList(testingCount));
                        
                        %% NEGATIVEDATA - REMOVE CURRENT USER DATA FROM trainNegHistogram%
                        trainNegData = trainNegHistogram;
                        trainNegData(userCount,:)=[];
                        trainNegHist = getFeatureData(trainNegData);
                        trainNegHist = trainNegHist(:,FeatureIndex);

                        expandedHistogram = [];
                        for dataCount = 1:size(trainNegHist,1)
                            %combined all histogram (expanded histogram)
                            expandedHistogram = [expandedHistogram;MixFeatureData(trainNegHist(dataCount,:))];
                        end
                        
                        trainNegHist = normc(expandedHistogram);
                        
                        %% NEGATIVEDATA - REMOVE CURRENT USER DATA FROM testNegHistogram
                        testNegData = testNegHistogram;
                        testNegData(userCount,:)=[];
                        testNegHist = getFeatureData(testNegData);
                        testNegHist = testNegHist(:,FeatureIndex);

                        expandedHistogram = [];
                        for dataCount = 1:size(testNegHist,1)
                            %combined all histogram (expanded histogram)
                            expandedHistogram = [expandedHistogram;MixFeatureData(testNegHist(dataCount,:))];
                        end
                        
                        testNegHist = normc(expandedHistogram);
                        
                        %% POSITIVEDATA - SET POSITIVE DATA for training and testing
                        trainPosData = trainHistogram_old;
                        testPosData = testHistogram;
                        
                        trainPosHist = getFeatureData(trainPosData);
                        trainPosHist = trainPosHist(:,FeatureIndex);

                        expandedHistogram = [];
                        for dataCount = 1:size(trainPosHist,1)
                            %combined all histogram (expanded histogram)
                            expandedHistogram = [expandedHistogram;MixFeatureData(trainPosHist(dataCount,:))];
                        end
                        
                        trainPosHist = normc(expandedHistogram);
                        
                        testPosHist = getFeatureData(testPosData);
                        testPosHist = testPosHist(:,FeatureIndex);

                        expandedHistogram = [];
                        for dataCount = 1:size(testPosHist,1)
                            %combined all histogram (expanded histogram)
                            expandedHistogram = [expandedHistogram;MixFeatureData(testPosHist(dataCount,:))];
                        end
                        
                        testPosHist = normc(expandedHistogram);
                        
                        testDataAns = [ones(numOfFlick*size(testPosData,1),1);zeros(numOfFlick*size(testNegData,1),1)];
                        
                        %% ======= TRAINING !!! ======
                        trainingTime1 = 0;
                        trainingTime2 = 0;
                        if (periodList(trainingCount)==periodList(testingCount))
                            % when current period is 3, calculate mean neg                                                        
                            t = clock;
                            [meanNeg{userCount}] = getMeanTrainingFeature2(trainNegHist,FeatureIndex);
                            trainingTime1 = etime(clock,t);
                            
                        end
                        
                        %calculate mean positive (new behavior)
                        t2 = clock;
                        [meanData_new] = getMeanTrainingFeature2(trainHistogram_new,FeatureIndex);
                        trainingTime2 = etime(clock,t2);
                        totalTIME_training{userCount,testingCount} = trainingTime1+trainingTime2;
                        
                        %% TESTING
                        t3 = clock;
                        [distanceTestWNewAll,distanceWithNegAll,distanceWithNegMultiAll,classResult,probability] = Nearest_Centroid_Classifier_Feature2 (FeatureIndex,testPosHist,testNegHist,meanData_new,meanNeg{userCount});
                        totalTIME_testing{userCount,testingCount} = etime(clock,t3);
                        
                        %% calculate FAR, FRR AND ACCURACY
                        thresholdtable = repmat(0:0.01:1,numel(testDataAns),1);
                        datatable = repmat(probability(:,1),1,101);
                        testingResult = datatable>thresholdtable;
                        
                        correctRate = [];
                        FAR = [];
                        FRR = [];
                        for i = 1:size(thresholdtable,2)
                            [ c, fa, Fr ] = MobileCorrectRate_FAR_FRR(testingResult(:,i),testDataAns,size(testPosHist, 1));   %?????FAR?FRR
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
                        
                        classTestPos = ones(numOfFlick*size(testPosData,1),1);
                        classTestNeg = zeros(numOfFlick*size(testNegData,1),1);
                        CP = classperf([classTestPos;classTestNeg], classResult);
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