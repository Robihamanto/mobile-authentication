clear;
addpath('..\0. otherImportantFunction');
folderNR = 'meeting0110\5Rounds60peoples\SingleModel\' ;
allAllavgeer=[];

for round=1:5
    folderN=[folderNR 'SingleModel' num2str(round)];
    posture = 'sit_long';
        
    %parameter setting
    numOfFlick = 5;
    % userInvolved = [1:70];
    userInvolved = [1:4 7:12 14:30 35 36 39:57 59:70];
%     userInvolved = [1 3 11 12 19 20 39 42 64 65];
    
    numOfNegData = 40;
    sampleset_path = ['..\..\1211_' num2str(numOfFlick) 'flicks_train_test_allperiod_v5_5round'];
    
    % 1=linearSVM 2=CART 3=NN 4=Bayse 5=WKNN
    classifierList = [1];
    
    FeatureIndex = [1:49];
    
    for classifierIndex = 1:numel(classifierList)
        classifierNum = classifierList(classifierIndex);
        [penaltyList,hiddenSizesNum] = ClassifierSetParameter(classifierNum);
        
        %% Load negative for training and testing
        load([sampleset_path '\' posture '\round_' num2str(round) '_train_neg_sampleSet.mat'], 'trainNegHistogram');
        load([sampleset_path '\' posture '\round_' num2str(round) '_test_neg_sampleSet.mat'], 'testNegHistogram');
        
        period =[3 4 5 6 7 8];
        for trainingCount = 1 : 1
            testingPeriod=period;
            avgAccu=[];
            for testingCount = 1 :numel(testingPeriod)
                correct_rate = [];
                Average_AR = [];
                Average_FAR = [];
                Average_FRR = [];
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
                    
                    trainNegData = trainNegData;
                    testNegData = testNegData;
                    
                    % Load positive for training
                    load([sampleset_path '\' posture '\user_' num2str(userID) '_period_' num2str(period(trainingCount)) '_round_' num2str(round) '_train_sampleSet.mat'], 'trainHistogram');
                    trainPosData = trainHistogram;
                    fprintf('\nTraining for user %d, period %d, no lopoUser \n',userID, period(trainingCount));
                    
                    if (trainingCount==testingCount)
                        % Load positive for training
                        load([sampleset_path '\' posture '\user_' num2str(userID) '_period_' num2str(period(trainingCount)-1) '_round_' num2str(round) '_train_sampleSet.mat'], 'trainHistogram');
                        trainPosData = [trainPosData;trainHistogram];
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
%save average EER for all ROUND
filename = ['..\exp_result\' folderNR '\averageEER'];
excelName = [filename '.xlsx'];
xlswrite(excelName,allAllavgeer,'AvgEER','A1');

