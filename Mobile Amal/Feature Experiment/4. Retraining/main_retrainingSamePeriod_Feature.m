clear;
addpath('..\0. otherImportantFunction');

%parameter setting
numOfFlick = 5;
userInvolved = [1:5 7:57 59:102];
numOfNegData = 40;
sampleset_path = ['..\0918_5flicks_100users_5round'];
sampleset_path_Neg = ['..\0918_5flicks_100users_5round'];

% 1=linearSVM 2=CART 3=NN 4=Bayse 5=WKNN
classifierList = [1];

FeatureIndex = [1:49];

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
    folderRoot = ['Retraining3_' posture];
    folderNR = ['Result final retraining feature Test\5rounds60peoples\' posture '\' folderRoot '\'] ;
    
    for round=1
        folderN=[folderNR 'Retrainingnotmix_' posture num2str(round)];
        folderToMake1 = ['..\exp_result\' folderN '\eer\eer_1\'];
        mkdir(folderToMake1);
        
        for classifierIndex = 1:numel(classifierList)
            classifierNum = classifierList(classifierIndex);
            [penaltyList,hiddenSizesNum] = ClassifierSetParameter(classifierNum);
            
            period =[3 4 5 6 7 8];
            for trainingCount = 1 : 1
                testingPeriod=period;
                avgAccu=[];
                for testingCount = 1 :numel(testingPeriod)
                    %% Load negative for training and testing
                    load([sampleset_path_Neg '\' posture '\round_' num2str(round) '_train_neg_sampleSet.mat'], 'trainNegHistogram');
                    load([sampleset_path_Neg '\' posture '\round_' num2str(round) '_test_neg_sampleSet.mat'], 'testNegHistogram');
                    %%
                    correct_rate = [];
                    Average_AR = [];
                    Average_FAR = [];
                    Average_FRR = [];
                    for userCount = 1:numel(userInvolved)
                        userID = userInvolved(userCount);
                        
                        %remove current user data from trainNegHistogram
                        trainNegData = trainNegHistogram;
                        trainNegData(userCount,:)=[];
                        trainNegFeature = getFeatureData(trainNegData);
                        
                        %remove current user data from testNegHistogram
                        testNegData = testNegHistogram;
                        testNegData(userCount,:)=[];
                        testNegFeature = getFeatureData(testNegData);
                        
                        % Load positive for training
                        load([sampleset_path '\' posture '\user_' num2str(userID) '_period_' num2str(period(testingCount)) '_round_' num2str(round) '_train_sampleSet.mat'], 'trainHistogram');
                        trainPosData = trainHistogram;
                        trainPosFeature = getFeatureData(trainPosData);
                        fprintf('\nTraining for user %d, period %d, no lopoUser \n',userID, period(trainingCount));
                        % Load positive for testing
                        load([sampleset_path '\' posture '\user_' num2str(userID) '_period_' num2str(period(testingCount))  '_round_' num2str(round) '_test_sampleSet.mat'], 'testHistogram');
                        testPosData = testHistogram;
                        testPosFeature = getFeatureData(testPosData);
                        allData = [trainPosFeature;trainNegFeature;testPosFeature;testNegFeature];
                        normalizeData = normc(allData);
                        
                        trainPosFeature = normalizeData(1:size(trainPosFeature,1),:);
                        trainNegFeature = normalizeData(size(trainPosFeature,1)+1:size(trainPosFeature,1)+size(trainNegFeature,1),:);
                        testPosFeature = normalizeData(size(trainPosFeature,1)+size(trainNegFeature,1)+1:size(trainPosFeature,1)+size(trainNegFeature,1)+size(testPosFeature,1),:);
                        testNegFeature = normalizeData(size(trainPosFeature,1)+size(trainNegFeature,1)+size(testPosFeature,1)+1:size(trainPosFeature,1)+size(trainNegFeature,1)+size(testPosFeature,1)+size(testNegFeature,1),:);
                        testDataAns = [ones(size(testPosData,1),1);zeros(size(testNegData,1),1)];
               
                        model = Training_featureData (FeatureIndex, trainPosFeature, trainNegFeature, classifierNum,penaltyList,hiddenSizesNum,numOfFlick);

                        %testing
                        fprintf('\nTesting for user %d, no lopoUser %d \n',userID);
                        [classResult,probability,testDataAns] = Testing_featureData (FeatureIndex,testPosFeature,testNegFeature,model,classifierNum,numOfFlick);

                        %% DISPLAY
                        allData = [trainPosFeature;trainNegFeature;testPosFeature;testNegFeature];
                        numberDimensions = 3;
                        % % Use PCA function to reduce dimension of The Raw Feature from 25 dimension to 3 dimension
                        [coef, score,~,~, explained] = pca(allData, 'NumComponents', numberDimensions);
                        
                        trainP = size(trainPosFeature,1);%150
                        trainN = size(trainNegFeature,1);%495
                        testNewP = size(testPosFeature,1);%150
                        testNewN = size(testNegFeature,1);%495
                        
                        figure
                        % train old positive
                        scatter3(score(1:trainP,1), ...
                                 score(1:trainP,2), ...
                                 score(1:trainP,3), 'Marker','^', 'MarkerFaceColor','blue')
                        set(gca,'XLim',[-0.3 0.3],'YLim',[-0.3 0.3],'ZLim',[-0.3 0.3])
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
                        %test new positive
                        scatter3(score(trainP+trainN+1:trainP+trainN+testNewP,1), ...
                                 score(trainP+trainN+1:trainP+trainN+testNewP,2), ...
                                 score(trainP+trainN+1:trainP+trainN+testNewP,3), 'Marker','^', 'MarkerFaceColor','green')
                        hold on
                        %test new negative
                        scatter3(score(trainP+trainN+testNewP+1:trainP+trainN+testNewP+testNewN,1), ...
                                 score(trainP+trainN+testNewP+1:trainP+trainN+testNewP+testNewN,2), ...
                                 score(trainP+trainN++testNewP+1:trainP+trainN+testNewP+testNewN,3), 'Marker','s', 'MarkerFaceColor','yellow')
                        saveas(gcf,sprintf('Figure user %d round %d.png' ,userTestCount,round));
                        
%%          
                        
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
                        
                        classTestPos = ones(size(testPosFeature,1),1);
                        classTestNeg = zeros(size(testNegFeature,1),1);
                        CP = classperf([classTestPos;classTestNeg], classResult);
                        correct_rate = [correct_rate;CP.CorrectRate];
                    end
                    
                    % save accuracy to file
                    filename = ['..\exp_result\' folderN '\accu_Train' num2str(period(trainingCount)) '_test' num2str(period(testingCount))];
                    excelName = [filename '.xlsx'];
                    xlswrite(excelName,correct_rate,'CorrectRate','A1');
                    avgAccu = [avgAccu mean(correct_rate)];
                    
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
%     save average EER for all ROUND
    filename = ['..\exp_result\' folderNR '\averageEER'];
    excelName = [filename '.xlsx'];
    xlswrite(excelName,allAllavgeer,'AvgEER','A1');
    
end