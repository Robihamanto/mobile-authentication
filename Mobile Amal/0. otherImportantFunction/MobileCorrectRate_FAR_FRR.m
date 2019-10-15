function  [ correctRate, FAR_rate, FRR_rate ] = MobileCorrectRate_FAR_FRR( testingResult,testingAns,numOfPos)

votingresult = sum( testingResult(testingAns == 1,:),2 ) > 0;
votingresult = [votingresult; (sum( testingResult(testingAns == 0,:),2 ) == 0)];

numOfTestingData = numel(testingAns);
numOfFRRcount = numel(find(votingresult(1:numOfPos) == 0));
numOfFARcount = numel(find(votingresult((numOfPos+1):end) == 0));
numOfcorrect = numel(find(votingresult(1:numOfPos) == 1)) + numel(find(votingresult((numOfPos+1):end) == 1));

correctRate = numOfcorrect / numOfTestingData;
FAR_rate = numOfFARcount / (numOfTestingData - numOfPos);
FRR_rate = numOfFRRcount / numOfPos;

end