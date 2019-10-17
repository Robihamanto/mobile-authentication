function [trainingPositive, testingPositive, trainingNegative, testingNegative] = preProcessData(usersInvolved, periods)
display('Start normalizing data, please wait..');
roundSize = 1;
sampleSetDataPath = '..\DataSet\New';
postures = {'sit_long'};

allData = [];

for postureString = postures
    posture = postureString{1};
    for round = 1:roundSize
        % Load Negative Data for Training & Testing
        load([sampleSetDataPath '\' posture '\round_' num2str(round) '_train_neg_sampleSet.mat']);
        load([sampleSetDataPath '\' posture '\round_' num2str(round) '_test_neg_sampleSet.mat']);
        negativeTrainingData = getFeatureDataV2(trainNegHistogram);
        negativeTestingData = getFeatureDataV2(testNegHistogram);
        allData = [negativeTrainingData; negativeTestingData];
        for period = periods
            display(['Normalizing data Period: ' num2str(period) ' of ' num2str(periods(numel(periods)))]);
            for userIndex = 1:numel(usersInvolved)
                user = usersInvolved(userIndex);
                load([sampleSetDataPath '\' posture '\' 'user_' num2str(user) '_period_' num2str(period) '_round_' num2str(round) '_train_sampleSet.mat']);
                load([sampleSetDataPath '\' posture '\' 'user_' num2str(user) '_period_' num2str(period) '_round_' num2str(round) '_test_sampleSet.mat']);
                positiveTrainingData = getFeatureDataV2(trainHistogram); trainHistogram = [];
                positiveTestingData = getFeatureDataV2(testHistogram); testHistogram = [];
                allData = [allData; positiveTrainingData; positiveTestingData];
            end
        end
    end
end

normalizedData = normc(allData);

trainingNegative = normalizedData(1:500,:);
testingNegative = normalizedData(501:1000,:);

trainingPositive = [];
testingPositive  = [];

for i = 1000:300:size(normalizedData,1)-300
    trainingPositive = [trainingPositive; normalizedData(i+1:i+150,:)];
    testingPositive  = [testingPositive;  normalizedData(i+151:i+300,:)];
end
display('Data normalizing complete.');
end
