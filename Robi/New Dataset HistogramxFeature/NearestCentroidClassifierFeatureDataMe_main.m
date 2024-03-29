clear; clc;
addpath('..\lib');

experimentMode = 'NCC with feature new me';
folderRoot = ['Experiment result ' experimentMode];
if ~exist(folderRoot, 'dir')
    mkdir(folderRoot);
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
        negativeTrainingData = getFeatureDataV2(trainNegHistogram);
        negativeTestingData = getFeatureDataV2(testNegHistogram);
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
                positiveTrainingData = getFeatureDataV2(trainHistogram); trainHistogram = [];
                positiveTestingData = getFeatureDataV2(testHistogram); testHistogram = [];
                
                negativeForTrainingData = negativeTrainingData;
                negativeForTrainingData(userIndex,:) = []; % Remove current user data from Negative Training Data
                negativeForTestingData = negativeTestingData;
                negativeForTestingData(userIndex,:) = []; % Remove current user data from Negative Testing Data
                
                dataIndex = [size(positiveTrainingData,1);size(positiveTestingData,1);size(negativeForTrainingData,1);size(negativeForTestingData,1);];
                tempData = [positiveTrainingData; positiveTestingData; negativeForTrainingData; negativeForTestingData];
                normalizedData = normc(tempData);
                
                positiveTrainingDataIndex       = dataIndex(1);
                positiveTestingDataIndex        = dataIndex(1) + dataIndex(2);
                negativeForTrainingDataIndex    = dataIndex(1) + dataIndex(2) + dataIndex(3);
                negativeForTestingDataIndex     = dataIndex(1) + dataIndex(2) + dataIndex(3) + dataIndex(4);
                
                positiveTrainingData        = normalizedData(1:positiveTrainingDataIndex,:);
                positiveTestingData         = normalizedData(positiveTrainingDataIndex+1:positiveTestingDataIndex,:);
                negativeForTrainingData     = normalizedData(positiveTestingDataIndex+1:negativeForTrainingDataIndex,:);
                negativeForTestingData      = normalizedData(negativeForTrainingDataIndex+1:negativeForTestingDataIndex,:);
                
                if period == 3
                    fprintf('++ Training period: %d Start \n', period);
                    meanNegativeData{user} = getMeanTrainingData(negativeForTrainingData, usedFeatureIndex);
                end
                
                meanPositiveData = getMeanTrainingData(positiveTrainingData, usedFeatureIndex);
                fprintf('>> Testing period: %d Start \n', period);
                [~,~,~,classResult,probability] = Nearest_Centroid_Classifier(usedFeatureIndex,positiveTestingData,negativeForTestingData,meanPositiveData,meanNegativeData{user});
                
                % Calculate FAR, FRR AND ACCURACY
                testDataAnswer = [ones(size(positiveTestingData,1),1); zeros(size(negativeForTestingData,1),1)];
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
                
                classTestPositive = ones(size(positiveTestingData,1),1);
                classTestNegative = zeros(size(negativeForTestingData,1),1);
                
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
    
    fileNameForAllAverageEER = [eerFolder ' All Average EER ' folderRoot '.xlsx'];
    xlswrite(fileNameForAllAverageEER, allAverageEER, 'EER', 'A1');
    fileNameForAllAverageAccuracy = [eerFolder ' All Average Accuracy ' folderRoot '.xlsx'];
    xlswrite(fileNameForAllAverageAccuracy, averageAccuracy, 'AverageAccuracy', 'A1');
    winopen(fileNameForAllAverageEER);
end

disp('Experiment completed.')











