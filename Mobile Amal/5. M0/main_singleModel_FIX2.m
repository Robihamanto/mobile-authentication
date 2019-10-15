clear;
addpath('..\0. otherImportantFunction');

%parameter setting
numOfFlick = 5;
% userInvolved = [1:4 7:12 14:30 35 36 39:57 59:70];
userInvolved = [1:5 7:57 59:102];
numOfNegData = 40;
% sampleset_path = ['..\..\0506_' num2str(numOfFlick) 'flicks_train_test_allperiod_5round'];
% sampleset_path_Neg = ['..\..\0506_' num2str(numOfFlick) 'flicks_train_test_allperiod_5round'];
sampleset_path = ['..\..\1211_' num2str(numOfFlick) 'flicks_train_test_allperiod_v5_5round'];
sampleset_path_Neg = ['..\Milan_NegData_EachPeriod'];

% 1=linearSVM 2=CART 3=NN 4=Bayes 5=WKNN
classifierList = [1];

FeatureIndex = [1:49];

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
    allAllavgeer=[];
    folderRoot = ['Single_' posture];
    folderNR = ['Result final v2\5rounds60peoples\' posture '\' folderRoot '\']
    
    for round=1
        folderN=[folderNR folderRoot num2str(round)];
        folderToMake1 = ['..\exp_result\' folderN '\eer\eer_1\'];
        %     folderToMake2 = ['..\exp_result\' folderN '\userData\4\'];
        %     folderToMake3 = ['..\exp_result\' folderN '\userData\5\'];
        %     folderToMake4 = ['..\exp_result\' folderN '\userData\6\'];
        %     folderToMake5 = ['..\exp_result\' folderN '\userData\7\'];
        %     folderToMake6 = ['..\exp_result\' folderN '\userData\8\'];
        mkdir(folderToMake1);
        %     mkdir(folderToMake2);
        %     mkdir(folderToMake3);
        %     mkdir(folderToMake4);
        %     mkdir(folderToMake5);
        %     mkdir(folderToMake6);
        
        for classifierIndex = 1:numel(classifierList)
            classifierNum = classifierList(classifierIndex);
            [penaltyList,hiddenSizesNum] = ClassifierSetParameter(classifierNum);
            
            period =[3 4 5 6 7 8];
            for trainingCount = 1 : 1
                testingPeriod=period;
                avgAccu=[];
                for testingCount = 1 :numel(testingPeriod)
                    correct_rate = [];
                    Average_AR = [];
                    Average_FAR = [];
                    Average_FRR = [];
                    %% Load negative for training and testing
%                     load([sampleset_path_Neg '\' posture '\round_' num2str(round) '_period_' num2str(period(testingCount)) '_train_neg_sampleSet.mat'], 'trainNegHistogram');
%                     load([sampleset_path_Neg '\' posture '\round_' num2str(round) '_period_' num2str(period(testingCount)) '_test_neg_sampleSet.mat'], 'testNegHistogram');
                    load([sampleset_path_Neg '\round_' num2str(round) '_period_' num2str(period(testingCount)) '_train_neg_sampleSet.mat'], 'trainNegHistogram');
                    load([sampleset_path_Neg '\round_' num2str(round) '_period_' num2str(period(testingCount)) '_test_neg_sampleSet.mat'], 'testNegHistogram');
                    %%
                    
                    for userCount = 1:numel(userInvolved)
                        userID = userInvolved(userCount);
                        
                        %remove current user data from trainNegHistogram
                        trainNegData= [];
                        rng(45);
                        randNum = randperm(size(trainNegHistogram, 1));
                        
                        for negUserIndex = 1:numOfNegData
                            if randNum(negUserIndex) ~= userCount
                                negData_perUser = trainNegHistogram(randNum(negUserIndex),:);
                                trainNegData = [trainNegData; negData_perUser];
                            end
                        end
                        
                        %remove current user data from testNegHistogram
                        testNegData= [];
                        rng(40);
                        randNum = randperm(size(testNegHistogram, 1));
                        
                        for negUserIndex = 1:numOfNegData
                            if randNum(negUserIndex) ~= userCount
                                negData_perUser = testNegHistogram(randNum(negUserIndex),:);
                                testNegData = [testNegData; negData_perUser];
                            end
                        end
                        
                        % Load positive for training
                        load([sampleset_path '\' posture '\user_' num2str(userID) '_period_' num2str(period(trainingCount)) '_round_' num2str(round) '_train_sampleSet.mat'], 'trainHistogram');
                        trainPosData = trainHistogram;
                        
                        fprintf('\nTraining for user %d, period %d, no lopoUser \n',userID, period(trainingCount));
                        
                        if (trainingCount==testingCount)
                            % Load additional positive for training
                            load([sampleset_path '\' posture '\user_' num2str(userID) '_period_' num2str(period(trainingCount)-1) '_round_' num2str(round) '_train_sampleSet.mat'], 'trainHistogram');
                            trainPosData = [trainPosData;trainHistogram];
                            % Load additional negative for training
%                             load([sampleset_path_Neg '\' posture '\round_' num2str(round) '_period_' num2str(period(testingCount)-1) '_train_neg_sampleSet.mat'], 'trainNegHistogram');
                            load([sampleset_path_Neg '\round_' num2str(round) '_period_' num2str(period(testingCount)-1) '_train_neg_sampleSet.mat'], 'trainNegHistogram');
                            trainNegDataTemp = trainNegHistogram;
                            trainNegDataT= [];
                            rng(45);
                            randNum = randperm(size(trainNegDataTemp, 1));
                            
                            for negUserIndex = 1:numOfNegData
                                if randNum(negUserIndex) ~= userCount
                                    negData_perUser = trainNegDataTemp(randNum(negUserIndex),:);
                                    trainNegDataT = [trainNegDataT; negData_perUser];
                                end
                            end
                            trainNegData = [trainNegData;trainNegDataT];
                            model(userCount) = Training_pureSVM (FeatureIndex, trainPosData, trainNegData, classifierNum,penaltyList,hiddenSizesNum);
                        end
                        
                        % Load positive for testing
                        load([sampleset_path '\' posture '\user_' num2str(userID) '_period_' num2str(period(testingCount))  '_round_' num2str(round) '_test_sampleSet.mat'], 'testHistogram');
                        testPosData = testHistogram;
                        testDataAns = [ones(size(testPosData,1),1);zeros(size(testNegData,1),1)];
                        %testing
                        fprintf('\nTesting for user %d, no lopoUser %d \n',userID);
                        
                        [classResult,probability] = Testing_pureSVM (FeatureIndex,testPosData,testNegData,model(userCount),classifierNum);
                        
                        %calculate FAR, FRR
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
                        
                        %                      if (period(trainingCount)~=period(testingCount))
                        %                         filename = ['..\exp_result\' folderN '\userData\' num2str(period(testingCount)) '\user' num2str(userID)];
                        %                         excelName = [filename '.xlsx'];
                        %                         xlswrite(excelName,classResult,'accuracy','B1');
                        %                     end
                        
                        % save result class peruser to file
                        % filename = ['..\exp_result\' resultTestData '\test_' num2str(period(testingCount))  '\User' num2str(userCount) 'th_single_train' num2str(period(trainingCount)) '_test' num2str(period(testingCount))];
                        %
                        % excelName = [filename '.xlsx'];
                        % xlswrite(excelName,probability,'accuracy','A1');
                        
                        classTestPos = ones(size(testPosData,1),1);
                        classTestNeg = zeros(size(testNegData,1),1);
                        CP = classperf([classTestPos;classTestNeg], classResult);
                        correct_rate = [correct_rate;CP.CorrectRate];
                    end
                    
                    % save accuracy to file
                    filename = ['..\exp_result\' folderN '\accu_Train' num2str(period(trainingCount)) '_test' num2str(period(testingCount))];
                    excelName = [filename '.xlsx'];
                    xlswrite(excelName,correct_rate,'CorrectRate','A1');
                    avgAccu = [avgAccu mean(correct_rate)]
                    
                    % save FAR and FRR to file
                    filename = ['..\exp_result\' folderN '\eer\eer_1\TLEER_Train' num2str(period(trainingCount)) '_test' num2str(period(testingCount))];
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
                % compute EER and save it
                [allavgeer]=ComputeInformation(folderN);
                allAllavgeer = [allAllavgeer;allavgeer];
                
                % save average accuracy to file
                filename = ['..\exp_result\' folderN '\averageAccu'];
                excelName = [filename '.xlsx'];
                xlswrite(excelName,avgAccu,'AvgAccu','A1');
            end
        end
    end
    % save average EER for all ROUND
    filename = ['..\exp_result\' folderNR '\averageEER'];
    excelName = [filename '.xlsx'];
    xlswrite(excelName,allAllavgeer,'AvgEER','A1');
end