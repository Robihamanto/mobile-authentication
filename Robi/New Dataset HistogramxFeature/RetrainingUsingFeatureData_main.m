clear; clc;
addpath('..\lib')

experimentMode = 'Retraining with feature';
folderRoot = ['Experiment result ' experimentMode];
if ~exist(folderRoot, 'dir')
    mkdir(folderRoot)
end

%PARAMETER SETTING
usersInvolved = [1:5 7:57 59:102];
roundSize = 1;
sampleSetDataPath = '..\DataSet\New';
usedFeatureIndex = [1:49];
classifierNumber = 1;
numOfFlick = 5;
periods = [3 4 5 6 7 8];

postures = {'sit_long'};

for postureString = postures
    posture = postureString{1};
    [postureName] = GetPostureName(posture);
    accuracyFolder = [folderRoot '\' postureName '\Accuracy'];
    eerFolder = [folderRoot '\' postureName '\EER'];
    allAverageEER = [];
    
    for round = 1:roundSize
        accuracyFolderPerRound = [accuracyFolder ' round ' num2str(round)];
        eerFolderPerRound = [eerFolder ' round ' num2str(round)];
        mkdir(accuracyFolderPerRound);
        mkdir([eerFolderPerRound '\eer_1']);
        
        % Load Negative Data for Training & Testing
        load([sampleSetDataPath '\' posture '\round_' num2str(round) '_train_neg_sampleSet.mat']);
        load([sampleSetDataPath '\' posture '\round_' num2str(round) '_test_neg_sampleSet.mat']);
        
        negativeTrainingData = getFeatureData(trainNegHistogram);
        negativeTestingData = getFeatureData(testNegHistogram);
        
        [penaltyList, hiddenSizeNumber] = ClassifierSetParameter(1);
        display(['Round: ' num2str(round)]);
        for period = periods
            
            accuracy        = [];
            averageAccuracy = [];
            correctRate     = [];
            averageAR       = [];
            averageFAR      = [];
            averageFRR      = [];
            
            for userIndex = 1:numel(usersInvolved)
                user = usersInvolved(userIndex);
                load([sampleSetDataPath '\' posture '\' 'user_' num2str(user) '_period_' num2str(period) '_round_' num2str(round) '_train_sampleSet.mat']);
                load([sampleSetDataPath '\' posture '\' 'user_' num2str(user) '_period_' num2str(period) '_round_' num2str(round) '_test_sampleSet.mat']);
                
                positiveTrainingData = getFeatureData(trainHistogram);
                positiveTestingData = getFeatureData(testHistogram);
                
                if period == 3
                    load([sampleSetDataPath '\' posture '\' 'user_' num2str(user) '_period_' num2str(2) '_round_' num2str(round) '_train_sampleSet.mat']);
                    positiveTrainingData = [positiveTrainingData ;getFeatureData(trainHistogram)];
                    negativeForTrainingData = negativeTrainingData;
                    negativeForTrainingData(userIndex,:) = []; % Remove current user data from Negative Training Data
                    fprintf('++ Training period: %d Start \n', period);
                    model(user) = Training(usedFeatureIndex, positiveTrainingData, negativeForTrainingData, classifierNumber, penaltyList, hiddenSizeNumber, numOfFlick);
                else
                    negativeForTrainingData = negativeTrainingData;
                    negativeForTrainingData(userIndex,:) = []; % Remove current user data from Negative Training Data
                    fprintf('++ Training period: %d Start \n', period);
                    model(user) = Training(usedFeatureIndex, positiveTrainingData, negativeForTrainingData, classifierNumber, penaltyList, hiddenSizeNumber, numOfFlick);
                end
                
                negativeForTestingData = negativeTestingData;
                negativeForTestingData(userIndex,:) = []; % Remove current user data from Negative Testing Data
                fprintf('>> Testing period: %d Start \n', period);
                [classResult, probability] = Testing(usedFeatureIndex, positiveTestingData, negativeForTestingData, model(user), classifierNumber, numOfFlick);
                
                % Calculate FAR, FRR AND ACCURACY
                testDataAnswer = [ones(numOfFlick * size(positiveTestingData,1),1); zeros(numOfFlick * size(negativeForTestingData,1),1)];
                thresholdtable = repmat(0:0.01:1,numel(testDataAnswer),1);
                datatable = repmat(probability(:,1),1,101);
                testingResult = datatable>thresholdtable;
                
                correctRate = [];
                FAR = [];
                FRR = [];
                
                for i = 1:size(thresholdtable,2)
                    [ c, fa, Fr ] = MobileCorrectRate_FAR_FRR(testingResult(:,i),testDataAnswer,size(positiveTestingData, 1));  % FAR,FRR,EER
                    correctRate = [correctRate c];
                    FAR = [FAR fa];
                    FRR = [FRR Fr];
                end
                
                averageAR = [averageAR; correctRate];
                averageFAR = [averageFAR; FAR];
                averageFRR = [averageFRR; FRR];
                
                classTestPositive = ones(numOfFlick * size(positiveTestingData,1),1);
                classTestNegative = zeros(numOfFlick * size(negativeForTestingData,1),1);
                
                CP = classperf([classTestPositive;classTestNegative], classResult);
                accuracy = [accuracy;CP.CorrectRate];
            end
            
            disp('Writing Accuracy, FAR, and FRR to file');
            fileNameForAccuracy = [accuracyFolderPerRound '\' experimentMode ' Accuracy Training ' posture ' .xlsx'];
            xlswrite(fileNameForAccuracy, accuracy, ['accuracy period' num2str(3) 'testperiod' num2str(period)], 'A1');
            xlswrite(fileNameForAccuracy, mean(accuracy), ['accuracy period' num2str(3) 'testperiod' num2str(period)], 'D1');
            averageAccuracy = [averageAccuracy mean(accuracy)];
            
            fileNameForEER = [eerFolderPerRound '\eer_1\' experimentMode ' EER train' num2str(3) 'test' num2str(period) '.xlsx'];
            xlswrite(fileNameForEER,averageAR, 'ACR', 'A1');
            xlswrite(fileNameForEER,averageFAR, 'FAR', 'A1');
            xlswrite(fileNameForEER,averageFRR, 'FRR', 'A1');
        end
        
        [averageEER] = ComputeInformation(eerFolderPerRound);
        allAverageEER = [allAverageEER; averageEER];
    end
    
    fileNameForAllAverageEER = [eerFolder ' All Average EER' folderRoot '.xlsx'];
    xlswrite(fileNameForAllAverageEER, allAverageEER, 'EER', 'A1');
    winopen(fileNameForAllAverageEER)
end

disp('Experiment completed.')











