function[allavgeer]= ComputeInformation(folderName)

dirpath = '.';
datadirpath = ['D:\Matlab\Mobile Authentication\Mobile Amal\exp_result\' folderName '\eer'];
% datadirpath = ['D:\MOBILE RESEARCH\Feature Experiment\exp_result\Result final v2\5rounds60peoples\sit_long\Single_sit_long\Single_sit_long1\eer'];

allFiles = dir(datadirpath);
allNames = { allFiles.name };

for fileIndex = 3:(numel(allNames))
    [str,remain]=strtok(allNames{fileIndex},'.');
    %     if isempty(remain)
    temp = [];
    peopleROC = [];
    [str,remain]=strtok(allNames{fileIndex},'_');
    while ~isempty(remain)
        [temp,remain]=strtok(remain,'_');
    end
    if ~isempty(temp)
        rountnum = str2double(temp);
        
        newfilepath = [datadirpath '\' allNames{fileIndex}];
        newallFiles = dir(newfilepath);
        newallNames = { newallFiles.name };
        
        totalEER = [];
        allavgeer=[];
        for dataIndex = 3:(numel(newallNames))
            
            thedatapaht = [newfilepath '\' newallNames{dataIndex}];
            thedataFAR = xlsread(thedatapaht,'FAR');
            thedataFRR = xlsread(thedatapaht,'FRR');
            
            proportion = thedataFAR ./ thedataFRR;
            proportion(isnan(proportion)) = 1;
            proportion = atan(proportion);
            proportion = proportion .* 180 ./ pi;
            
            angle = [0:1:89];
            
            total_avg_far = [];
            total_avg_frr = [];
            rocCurve = [];
            avg_far = [];
            avg_frr = [];
            eer = [];
            peopleavgeer=[];
            tempavgeer = [];
            time = size(thedataFAR,1);
            peopleNum = time / rountnum;
            %         ==================================================
            
            for peopleindex = 1:size(thedataFAR,1)
                sortarray = [thedataFAR(peopleindex,:);thedataFRR(peopleindex,:);proportion(peopleindex,:)];
                sortarray = sortrows(sortarray',3)';
                sortfar = sortarray(1,:);
                sortfrr = sortarray(2,:);
                sortangle = sortarray(3,:);
                
                people_avg_far = [min(sortfar(1,sortangle(1,:)==angle(1)))];
                people_avg_frr = [min(sortfrr(1,sortangle(1,:)==angle(1)))];
                
                
                if(length(people_avg_far) == 0 )
                    people_avg_far = [0];
                    people_avg_frr = [1];
                end
                
                for index = 2:size(angle,2)
                    [~,numindex] = min(abs(sortangle(1,:)-angle(index)));
                    
                    
                    if(sortangle(1,numindex) > 45)
                        [minfar,ind] = min(sortfar(1,sortangle(1,:)==sortangle(1,numindex)));
                        ind = ind-1+numindex;
                        minfrr = sortfrr(1,ind);
                    else
                        [minfrr,ind] = min(sortfrr(1,sortangle(1,:)==sortangle(1,numindex)));
                        ind = ind-1+numindex;
                        minfar = sortfar(1,ind);
                    end
                    
                    if(sortangle(1,numindex)==angle(index))
                        people_avg_far = [people_avg_far minfar];
                        people_avg_frr = [people_avg_frr minfrr];
                    else
                        if(sortangle(1,numindex)<angle(index))
                            temp = sortangle(1,sortangle(1,:)>angle(index));
                            [~,tempindex] = min(abs(temp(1,:)-angle(index)));
                            
                            if(temp(1,tempindex) > 45)
                                [secfar,ind] = min(sortfar(1,sortangle(1,:)==temp(1,tempindex)));
                                secfrr = sortfrr(1,sortfar(1,:)==secfar & sortangle(1,:)==temp(1,tempindex));
                                secfrr = secfrr(1,1);
                            elseif(length(temp) == 0)
                                secfar = 1;
                                secfrr = 0;
                            else
                                [secfrr,ind] = min(sortfrr(1,sortangle(1,:)==temp(1,tempindex)));
                                secfar = sortfar(1,sortfrr(1,:)==secfrr & sortangle(1,:)==temp(1,tempindex));
                                secfar = secfar(1,1);
                            end
                            
                        else
                            temp = sortangle(1,sortangle(1,:)<angle(index));
                            if(length(temp) == 0 )
                                secfar = 0;
                                secfrr = 1;
                            else
                                [~,tempindex] = min(abs(temp(1,:)-angle(index)));
                                if(temp(1,tempindex) > 45)
                                    [secfar,ind] = min(sortfar(1,sortangle(1,:)==temp(1,tempindex)));
                                    secfrr = sortfrr(1,sortfar(1,:)==secfar & sortangle(1,:)==temp(1,tempindex));
                                    secfrr = secfrr(1,1);
                                else
                                    [secfrr,ind] = min(sortfrr(1,sortangle(1,:)==temp(1,tempindex)));
                                    secfar = sortfar(1,sortfrr(1,:)==secfrr & sortangle(1,:)==temp(1,tempindex));
                                    secfar = secfar(1,1);
                                end
                            end
                        end
                        
                        if(isnan(secfar) && isnan(secfrr))
                            thefar = nan;
                            thefrr = nan;
                        else
                            thefrr = ( ((secfrr-minfrr)*minfar )-((secfar-minfar)*minfrr) ) / ( ((secfrr-minfrr)*tan(angle(index)*pi/180)) - (secfar-minfar));
                            thefar = thefrr * tan(angle(index)*pi/180);
                        end
                        
                        people_avg_far = [people_avg_far thefar];
                        people_avg_frr = [people_avg_frr thefrr];
                    end
                end
                if(length(sortfar(1,sortangle(1,:) == 90 )) == 0)
                    people_avg_far = [people_avg_far 1];
                    people_avg_frr = [people_avg_frr 0];
                else
                    people_avg_far = [people_avg_far min(sortfar(1,sortangle(1,:) == 90 ))];
                    people_avg_frr = [people_avg_frr min(sortfrr(1,sortangle(1,:) == 90 ))];
                end
                if (people_avg_far(1,90)==0) && (people_avg_far(1,91)==1)
                    people_avg_far(1:91)=0;
                end
                
                total_avg_far = [total_avg_far; people_avg_far];
                total_avg_frr = [total_avg_frr; people_avg_frr];
                rocCurve = [rocCurve;people_avg_far;people_avg_frr];
            end
            temp_people = [nanmean(total_avg_far,1);nanmean(total_avg_frr,1)];
            
            %         ==================================================
            
            for timeIndex = 1:time
                
                avgeer = total_avg_far(timeIndex,46);
                
                eer = [eer;avgeer];
                tempavgeer = [tempavgeer;avgeer];
                if size(tempavgeer,1) == peopleNum
                    peopleavgeer = [peopleavgeer;mean(tempavgeer)];
                    tempavgeer = [];
                elseif numel(tempavgeer) > peopleNum
                    error('over tempavgeer');
                end
                
            end
            
            totalEER = [totalEER eer];
            pause(0.001);
            A1 = {'EER'};
            xlswrite(thedatapaht,A1,'EER','A1');
            xlswrite(thedatapaht,eer,'EER','B1');
            xlswrite(thedatapaht,rocCurve,'ROC','A1');
            xlswrite(thedatapaht,temp,'Temp','A1');
            allavgeer = [allavgeer peopleavgeer];
            peopleROC = [peopleROC;temp_people];
        end
        xlswrite([datadirpath '\totalEER.xlsx'],allavgeer,str,'A1');
        xlswrite([datadirpath '\totalEER.xlsx'],peopleROC,['zROC_' str],'A1');
    end
end
end