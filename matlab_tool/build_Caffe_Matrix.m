function build_Caffe_Matrix(model_name,Age,Layer,partition_mode, feature_sample_mode,featureType)
if ispc
	addpath('Z:\tzeng\ABA\automatedGeneLabelingProject\code');
else
	addpath('/home/tzeng/ABA/automatedGeneLabelingProject/code');
end
if nargin <5
    partition_mode= 'SagiPartition';
    feature_sample_mode ='max';
end
label_dir =translatePath('Z:\ABA\automatedGeneLabelingProject\manannot\data\labels');
data_dir=translatePath('Z:\ABA\autoGenelable_multi_lables_proj\data');

%bow_mex_file=fullfile(data_dir,['BOW_Matrix_' Age '.mat']);
serialDirs = get_serial_section_dirs(Age);

if strcmp(featureType,'singleImage')
	if strcmp(partition_mode,'SagiPartition') 
		%if finetune
			caffeFiles_s=['All_Section_7' '_L' num2str(Layer) '_'  partition_mode '_' feature_sample_mode '_' model_name '.mat'];
			caffe_mex_file=fullfile(data_dir,['Caffe_7'  partition_mode '_' lower(feature_sample_mode) '_Matrix_' Age '_L_' num2str(Layer) '_' model_name '.mat'])
		%else
		%	saveFile=fullfile(subDir,['All_Section_7' '_L' num2str(Layer) '_'  part_mode '_' feature_mode '_Caffe_VGG_FF.mat']);
		%end
	else
		%if finetune
		   %saveFile=fullfile(subDir,['All_Section'  '_L' num2str(Layer) '_'  part_mode '_' feature_mode '_' model_name '.mat']);
		   caffeFiles_s=['All_Section' '_L' num2str(Layer) '_'  partition_mode '_' feature_sample_mode '_' model_name '.mat'];
		   caffe_mex_file=fullfile(data_dir,['Caffe_'  partition_mode '_' lower(feature_sample_mode) '_Matrix_' Age '_L_' num2str(Layer) '_' model_name '.mat'])
		%else
		%	saveFile=fullfile(subDir,['All_Section'  '_L' num2str(Layer) '_'  part_mode '_' feature_mode '_Caffe_VGG_FF.mat']);
		%end
	end
elseif strcmp(featureType,'multiImage') 
	caffeFiles_s=['All_Section' '_L' num2str(Layer) '_' model_name '_.mat'];
	caffe_mex_file=fullfile(data_dir,['Caffe_Matrix_' Age '_L_' num2str(Layer) '_' model_name '.mat']);
elseif strcmp(featureType,'pairedImage')
    %caffeFiles_s=['All_Section' '_L' num2str(Layer) '_' model_name '_.mat'];
	%caffe_mex_file=fullfile(data_dir,['Caffe_Matrix_' Age '_L_' num2str(Layer) '_' model_name '.mat']);
	caffeFiles_s=['All_Section' '_L' num2str(Layer) '_'  partition_mode '_' feature_sample_mode '_' model_name '.mat'];
	caffe_mex_file=fullfile(data_dir,['Caffe_'  partition_mode '_' lower(feature_sample_mode) '_Matrix_' Age '_L_' num2str(Layer) '_' model_name '.mat'])
end




% if strcmpi(partition_mode,'SagiPartition')
    
    % %bowFile=fullfile(serialDirs{1},'All_Section_7Segi_partition_Mean_BOW.mat');
    % %bowFile_s=['All_Section_7'  partition_mode '_' lower(feature_sample_mode) '_overfeatFeature.mat'];
	% if finetune
    % caffeFiles_s=['All_Section_7' '_L' num2str(Layer) '_'  partition_mode '_' feature_sample_mode '_FineTune_Caffe_VGG_FF.mat'];
	% caffe_mex_file=fullfile(data_dir,['Caffe_7'  partition_mode '_' lower(feature_sample_mode) '_Matrix_' Age '_L_' num2str(Layer) '_FineTune.mat']);
	% else
	 % caffe_mex_file=fullfile(data_dir,['Caffe_7'  partition_mode '_' lower(feature_sample_mode) '_Matrix_' Age '_L_' num2str(Layer) '.mat']);
	 % caffeFiles_s=['All_Section_7' '_L' num2str(Layer) '_'  partition_mode '_' feature_sample_mode '_Caffe_VGG_FF.mat'];
	% end
% else 
    % %caffe_mex_file=fullfile(data_dir,['Caffe_'  partition_mode '_' lower(feature_sample_mode) '_Matrix_' Age '_L_' num2str(Layer) '.mat']);
    % %bowFile_s=['All_Section_'  partition_mode '_' lower(feature_sample_mode) '_overfeatFeature.mat'];
    % %caffeFiles_s=['All_Section' '_L' num2str(Layer) '_'  partition_mode '_' feature_sample_mode '_Caffe_VGG_FF.mat'];
	
	% if finetune
    % caffeFiles_s=['All_Section' '_L' num2str(Layer) '_'  partition_mode '_' feature_sample_mode '_FineTune_Caffe_VGG_FF.mat'];
	% caffe_mex_file=fullfile(data_dir,['Caffe_'  partition_mode '_' lower(feature_sample_mode) '_Matrix_' Age '_L_' num2str(Layer) '_FineTune.mat']);
	% else
	 % caffe_mex_file=fullfile(data_dir,['Caffe_'  partition_mode '_' lower(feature_sample_mode) '_Matrix_' Age '_L_' num2str(Layer) '.mat']);
	 % caffeFiles_s=['All_Section' '_L' num2str(Layer) '_'  partition_mode '_' feature_sample_mode '_Caffe_VGG_FF.mat'];
	% end
    
% end


if exist(caffe_mex_file,'file')==2
 disp(['File matrix : ' caffe_mex_file ' already exist quit ...']);
 return
 end
%initMatlabPool();

%bowFile=fullfile(serialDirs{1},'All_Section_BOW.mat');
%bowFile=fullfile(serialDirs{1},'All_Section_7Segi_partition_Mean_BOW.mat');
caffeFile=fullfile(serialDirs{1}, caffeFiles_s);
load(caffeFile);
%Vector=[ALL_SECTION_BOW{:}];
vlen=size(ALL_SECTION_OFF,1);

numDir=length(serialDirs);

CAFFE_Matrix=zeros(numDir,vlen);
c =0;
for i=1:length(serialDirs)
    cur_dir = serialDirs{i};
    if ~exist(cur_dir,'dir')
        continue;
    end
    c=c+1;
    disp(['process ' num2str(i) ' out of ' num2str(length(serialDirs)) ' files']);
    caffeFile=fullfile(cur_dir,caffeFiles_s);
    load(caffeFile);
    CAFFE_Matrix(c,:)=ALL_SECTION_OFF;
    if(sum(ALL_SECTION_OFF(:)~=0)==0)
		disp('ALL_SECTION_OFF are Zeros');
		disp(caffeFile);
	end
    serialNum=extrat_SectionNameFromPath(cur_dir);
    label_file=fullfile(label_dir,[serialNum '.mat']);
    load(label_file);
    labels{c}={MLevel.labels};
    dataSetsID(c)=str2num(serialNum);
    if i==1
       structureName ={MLevel.structInfo};
    end
    
end
CAFFE_Matrix=CAFFE_Matrix(1:c,:);
save( caffe_mex_file,'CAFFE_Matrix','labels', 'dataSetsID', 'structureName','-v7.3');
end
function str=extrat_SectionNameFromPath(subdir)
   idx= strfind(subdir,filesep);
   str=subdir(idx(end)+1:end);
   %str=subdir(idx(end-1)+1:end-1);
   str=str(1:9);
end

