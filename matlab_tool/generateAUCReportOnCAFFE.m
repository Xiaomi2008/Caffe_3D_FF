function generateAUCReportOnCAFFE(classfied_file)
% generateAUCReport(Age,partition_mode,feature_smaple_mode)partition_mode
%   input: 
%    Age, deleveloping stage,
%     partition_mode  ,  How to group experimental serial sections ,
%     defualt: SagiPartition -   subgroup into 7 saggital partitions by
%               evenly partitional images base on thier posotions
%               whole  -- subroup into just 1 by averaging or sum up image's
%               descriptors.
%     feature_sample_node:  mean or sum of each image's descriptors,
%     defulat - mean,     mean and sum

resultDir=translatePath('Z:\ABA\autoGenelable_multi_lables_proj\result');
levels=[1 2 3 4 5 6 7 8 9 10 11 12];
%levels=[1];
% if strcmpi(partition_mode,'sagipartition')
    % %file_part1='Sig7Partition';
    % file_part1='7SagiPartition';
    % if strcmpi(feature_smaple_mode,'mean')
        % file_part2='Mean';
    % else strcmpi(feature_smaple_mode,'Max')
        % file_part2='Max';
    % end
% elseif strcmpi(partition_mode,'whole')
    % file_part1='whole';
     % if strcmpi(feature_smaple_mode,'mean')
        % file_part2='mean';
    % else strcmpi(feature_smaple_mode,'Max')
        % file_part2='Max';
    % end
    
% end
c=0;
Mdata=zeros(length(levels),6);
for i=7:7
    %[saveFilePrefix '_ontologyL' num2str(i) '.mat']
    file=[classfied_file  '_ontologyL' num2str(levels(i)) '.mat' ];
    file=fullfile(resultDir,file);
    if exist(file,'file')
        load(file);
        %c=c+1;
        Mdata(i,1)=mean(pattern.auc);
        Mdata(i,2)=std(pattern.auc);
        Mdata(i,3)=mean(intensity.auc);
        Mdata(i,4)=std(intensity.auc);
        Mdata(i,5)=mean(density.auc);
        Mdata(i,6)=std(density.auc);
    end
end

reportFile= fullfile(resultDir,['auc_report_'  classfied_file '.csv']);

%if ~exist(reportFile,'file')
    fileID =fopen(reportFile,'w');
    fprintf(fileID,'%15s,%15s,%15s,%15s,%15s,%15s\n','pattern Mean AUC','pattern AUC STD','intensity Mean AUC','intensity AUC STD','density Mean AUC','density AUC STD');
    fprintf(fileID,'%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f\n',Mdata');
    fclose(fileID);
%end


end

