clear; clc;
addpath('..\lib');

experimentMode = 'NCC with feature and preprocessing';
folderRoot = ['Experiment result ' experimentMode];
if ~exist(folderRoot, 'dir')
    mkdir(folderRoot);
end

%PARAMETER SETTING
%usersInvolved = [1:5 7:57 59:102];
usersInvolved = [1:5 7:57 59:102];
roundSize = 1;
sampleSetDataPath = '..\DataSet\New';
usedFeatureIndex = [1:49];
classifierNumber = 1;
numOfFlick = 5;
periods = [3 4 5 6 7 8];
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
postures = {'sit_long'};

[trainingPositive, testingPositive, trainingNegative, testingNegative] = preProcessData(usersInvolved, periods);

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
        negativeTrainingData = trainingNegative;
        negativeTestingData = testingNegative;
        display(['Round: ' num2str(round)]);
        
        for periodIndex = 1:numel(periods)
            period = periods(periodIndex);
            accuracy        = [];
            averageAccuracy = [];
            correctRate     = [];
            averageAR       = [];
            averageFAR      = [];
            averageFRR      = [];
            
            for userIndex = 1:numel(usersInvolved)
                
                startDataIndex = ((periodIndex-1) * numel(usersInvolved) * 150) + ((userIndex-1)*150) + 1;
                endDataIndex = ((periodIndex-1) * numel(usersInvolved) * 150) + ((userIndex-1)*150) + 150;
                
                positiveTrainingData = trainingPositive(startDataIndex:endDataIndex,:);
                positiveTestingData = testingPositive(startDataIndex:endDataIndex,:);
                
                startDataIndex = (userIndex-1)*5+1;
                endDataIndex = (userIndex-1)*5+5;
                
                negativeForTrainingData = negativeTrainingData;
                negativeForTrainingData(startDataIndex:endDataIndex,:) = []; % Remove current user data from Negative Training Data
                negativeForTestingData = negativeTestingData;
                negativeForTestingData(startDataIndex:endDataIndex,:) = []; % Remove current user data from Negative Testing Data

                
                if period == 3
                    fprintf('++ Training period: %d Start \n', period);
                    meanNegativeData{userIndex} = getMeanTrainingData(negativeForTrainingData, usedFeatureIndex);
                end
                
                meanPositiveData = getMeanTrainingData(positiveTrainingData, usedFeatureIndex);
                fprintf('>> Testing period: %d User: %d Start \n', period, usersInvolved(userIndex));
                [~,~,~,classResult,probability] = Nearest_Centroid_Classifier(usedFeatureIndex,positiveTestingData,negativeForTestingData,meanPositiveData,meanNegativeData{userIndex});
                
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

disp('Experiment Completed');












