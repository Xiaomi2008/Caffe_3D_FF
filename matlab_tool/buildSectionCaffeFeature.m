function buildSectionCaffeFeature(ImageType,model_name,subDir,Layer,feature_mode,part_mode)
% 
% function : build combined features from a serial secction images (subDir) , saved it to one single file in each section images folder.
% 


%dirStruct =dir(subDir);
%dir_files =dirStruct(~[ dirStruct.isdir]);
%bow_files_idx = cellfun(@(x) ~isempty(findstr(x,'BOW.mat')),dir_files);
%bow_files=dir_files(bow_files_idx);
if nargin<5
    part_mode ='whole';
end

if nargin<4
    part_mode ='whole';
    feature_mode='max';
end

if ~strcmp(part_mode,'SagiPartition') && ~strcmp(part_mode,'whole')
    disp('part_mode must be either "SagiPartition" or "whole"' );
    return;
end


%saveFile=fullfile(subDir,'All_Section_7Segi_partition_Mean_BOW.mat');

if strcmp(part_mode,'SagiPartition')
    %if finetune
		saveFile=fullfile(subDir,['All_Section_7' '_L' num2str(Layer) '_'  part_mode '_' feature_mode '_' model_name '.mat']);
	%else
	%	saveFile=fullfile(subDir,['All_Section_7' '_L' num2str(Layer) '_'  part_mode '_' feature_mode '_Caffe_VGG_FF.mat']);
	%end
else
    %if finetune
	   saveFile=fullfile(subDir,['All_Section'  '_L' num2str(Layer) '_'  part_mode '_' feature_mode '_' model_name '.mat']);
	%else
	%	saveFile=fullfile(subDir,['All_Section'  '_L' num2str(Layer) '_'  part_mode '_' feature_mode '_Caffe_VGG_FF.mat']);
	%end
end
% if exist(saveFile ,'file')==2
   % return;
% end
disp(['working on Dir ' subDir]);
fID = fopen(fullfile(subDir,'age.txt'));
age=textscan(fID,'%s');
fclose(fID);
age =char(age{1});
pos_range_struct =partition_filesPos_inSection( subDir,age,part_mode);
for i=1:length(pos_range_struct)
    sectionFiles=pos_range_struct{i}.pos_section_files;
    
	featureFiles=findExistingFeatureFiles(ImageType,model_name,subDir,sectionFiles,Layer);
	numFiles=length(featureFiles);
    
    
    if numFiles >0
         %overfeatFeaturefile=[num2str(sectionFiles(1)) '_' num2str(imSize(1)) 'X' num2str(imSize(2)) '_L' num2str(Layer) '_overfeatFeature.mat'];
		 %caffe_vgg_feature_file=[num2str(sectionFiles(j)) '_VGG_16_L_' num2str(Layer) '_caffeFeature_.mat'];
         %zeroDescFile=[num2str(sectionFiles(1)) '_ZeroDes.mat'];
         load(featureFiles{1});
		 featureOut=featureOut(:);
         %load(fullfile(subDir,zeroDescFile));
         %s_length=length(SCALE);
         %for s=1:s_length
             %T_OFF{s}=[];
         %end
        T_OFF =zeros(length(numFiles),size(featureOut,1));
		T_OFF(1,:)=featureOut;
        for j=2:numFiles
             %caffe_vgg_feature_file=[num2str(sectionFiles(j)) '_VGG_16_L_' num2str(Layer) '_caffeFeature_.mat'];
            %zeroDescFile=[num2str(sectionFiles(1)) '_ZeroDes.mat'];
            %bowfile=fullfile(subDir,bowfile);
            load(featureFiles{j});
			   
				 try 
                 T_OFF(j,:)=featureOut(:);
				 catch e
				    size(T_OFF)
				    size(featureOut(:))
					disp(['File size mismatched  :  ' fullfile(subDir, caffe_vgg_feature_file) ]);
					return
					%imgfile =fullfile(subDir,[num2str(sectionFiles(j)) '.jpg']);
					%[featureOut, mapsn, mapsize] =extractAndSaveOverFeatFeature(   sectionFiles(j),[231 231],8);
					%T_OFF(j,:)=featureOut;
				end
        end
    end
    
    if numFiles ==0
        switch Layer
		    case { 'pool1'}
			    vectorSize =112*112*64;
			case{'conv2_1','conv2_2'}
			     vectorSize =112*112*128;
		    case { 'pool2'}
			    vectorSize =56*56*128
			case{'conv3_1','conv3_2','conv3_3'}
			    vectorSize =56*56*256;
		     case {'pool3'}
			    vectorSize =28*28*256;
			 case{'conv4_1','conv4_2','conv4_3'}
                vectorSize =28*28*512;
            case {'pool4','conv5_1','conv5_2','conv5_3'}
                vectorSize =14*14*512;
			case{'pool5'}
			    vectorSize=7*7*512;
			case{'fc6','fc7'}
			    vectorSize=4096;
			   
        end
            T_OFF(i,:)=zeros(1,vectorSize);
    end
    % do average or sum on the saggital partion(total 7) on the bow with in
    % each partion, total =7*500k+3scale
    %for s=1:s_length
   % try
          % disp(['feature_mode is '  feature_mode]);
            [r,c]=size(T_OFF);
            if r==1 || c==1
               ALL_SECTION_OFF(i,:) =T_OFF;
            else
                if strcmp(feature_mode,'max')
				    disp('max computeted ...');
                    ALL_SECTION_OFF(i,:)=max(T_OFF); 
                elseif strcmp(feature_mode,'sum')
                    ALL_SECTION_OFF(i,:)=sum(T_OFF);
                elseif strcmp(feature_mode,'mean')
				    disp('mean computeted ...');
                    ALL_SECTION_OFF(i,:)=mean(T_OFF,1);
                end
            end
    %catch e
     %   ix =0;
   % end
   
    
end
ALL_SECTION_OFF=reshape(ALL_SECTION_OFF,numel(ALL_SECTION_OFF),1);
save(saveFile,'-V7', 'ALL_SECTION_OFF','pos_range_struct','age');
%disp(['saving ' saveFile '...']);

end


function featureFiles=findExistingFeatureFiles(ImageType,model_name,subDir,sectionIds,Layer)
    count =0;
	
	%101085821_VGG_16_L_conv4_2_FineTune_caffeFeature_.mat
	for i=1:length(sectionIds)
	    % if finetune
		   % vgg_feature_file=[num2str(sectionIds(i)) '_VGG_16_L_' num2str(Layer) '_FineTune_caffeFeature_.mat'];
		   
		% else
		   % vgg_feature_file=[num2str(sectionIds(i)) '_VGG_16_L_' num2str(Layer) '_caffeFeature_.mat'];
		% end
		 if strcmp(ImageType,'ISH')
			im_name =num2str(sectionIds(i));
		 elseif strcmp(ImageType,'MSK')
			im_name =[num2str(sectionIds(i)) '_msked'];
		 end
		 featureMatFile_suffix =[model_name '_' Layer];
		 vgg_feature_file=[im_name '_' featureMatFile_suffix '_.mat']; 
		
		vgg_feature_file =fullfile(subDir,  vgg_feature_file);
		if exist(vgg_feature_file,'file') ==2
		  count=count+1;
		  featureFiles{count}=vgg_feature_file;
		end
	end
end