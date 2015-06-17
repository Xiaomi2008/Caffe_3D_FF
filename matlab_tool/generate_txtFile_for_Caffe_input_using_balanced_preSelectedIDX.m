function generate_txtFile_for_Caffe_input_using_balanced_preSelectedIDX( age, numOfImagesForEachSerial,ontology_level,mask_image,classType)
%[level_labs,bowmatrix]=findFirstConnatAnnotatedLabel(level_labs_pattern,norm_matrix);

if nargin <4
 mask_image =false;
 end
addpath(translatePath('Z:\ABA\automatedGeneLabelingProject\code'));
data_dir=translatePath('Z:\ABA\automatedGeneLabelingProject\manannot\data');
balanced_BOW_matrix_pattern_file    =fullfile(data_dir,['balanced_BOW_matrix_Age_' age 'OntologyLV_' num2str(ontology_level) '_pattern.mat']);
load(balanced_BOW_matrix_pattern_file);
p_train_IDX=train_IDX;
p_test_IDX=test_IDX;


[serial_section_info ,max_sagital_sectionPos]= getSerial_Info_ByAge(age);
%num_serail=length(serial_section_info);

idx=findFirstCannotAnnotatedLabel(serial_section_info,ontology_level);


data_dir_cur=translatePath('Z:\ABA\autoGenelable_multi_lables_proj\data');
filename =['train_validataion_test_set_for_caffeCNN_level' num2str(ontology_level) '_' age '.mat'];
filename=[data_dir_cur filesep filename];
pos_interval=max_sagital_sectionPos/(numOfImagesForEachSerial-1);
pos_interval_all=0:pos_interval:max_sagital_sectionPos;
pos_interval_all(1)=1;
val_ratio=0.5;
if exist(filename,'file')==2
    load(filename);
else
%     perm_serial_idx=randperm(num_serail);
%     t_n = round(train_ratio*num_serail);
%     v_n = round(val_ratio*num_serail);
%     train_serial = serial_section_info(perm_serial_idx(1:t_n));
%     val_serial   = serial_section_info(perm_serial_idx(t_n+1:t_n+v_n));
%     test_serial  = serial_section_info(perm_serial_idx(t_n+v_n+1:end));
%     save(filename,'train_serial','val_serial','test_serial');

  serial_section_info_remain       = serial_section_info(idx);
  train_serial                     = serial_section_info_remain(p_train_IDX);
  all_test_serial                  = serial_section_info_remain(p_test_IDX);
  test_len =length(all_test_serial);
  v_n      =round(val_ratio*test_len);
  val_serial                      = all_test_serial(1:v_n);
  test_serial                     = all_test_serial(v_n+1:end);
  remain_serial                   = serial_section_info(~idx);
  
  save(filename,'train_serial','val_serial','test_serial','all_test_serial','remain_serial');
end
  
 if mask_image
	train_output_file =     [data_dir_cur filesep 'train_msk_' age '.txt'];
	val_output_file   =     [data_dir_cur filesep 'val_msk_' age '.txt'];
	test_output_file  =     [data_dir_cur filesep 'test_msk_' age '.txt'];
	all_test_output_file  =     [data_dir_cur filesep 'all_test_msk_' age '.txt'];
    
    train_multiS_output_file =     [data_dir_cur filesep 'train_slice_msk_' age '.txt'];
    val_multiS_output_file   =     [data_dir_cur filesep 'val_slice_msk_' age '.txt'];
    test_multiS_output_file  =     [data_dir_cur filesep 'test_slice_msk_' age '.txt'];
    all_test_multiS_output_file  =     [data_dir_cur filesep 'all_test_slice_msk_' age '.txt'];
else
	train_output_file =     [data_dir_cur filesep 'train_ish_' age '.txt'];
	val_output_file   =     [data_dir_cur filesep 'val_ish_' age '.txt'];
	test_output_file  =     [data_dir_cur filesep 'test_ish_' age '.txt'];
	all_test_output_file  =     [data_dir_cur filesep 'all_test_ish_' age '.txt'];
    
    
    train_multiS_output_file =     [data_dir_cur filesep 'train_slice_ish_' age '.txt'];
    val_multiS_output_file   =     [data_dir_cur filesep 'val_slice_ish_' age '.txt'];
    test_multiS_output_file  =     [data_dir_cur filesep 'test_slice_ish_' age '.txt'];
	all_test_multiS_output_file  =     [data_dir_cur filesep 'all_test_slice_ish_' age '.txt'];
end



outputTxt(train_output_file,train_serial,pos_interval_all,mask_image,classType);
outputTxt(val_output_file,val_serial,pos_interval_all,mask_image,classType);
outputTxt(test_output_file,test_serial,pos_interval_all,mask_image,classType);
outputTxt(all_test_output_file,all_test_serial,pos_interval_all,mask_image,classType);


multiSlices_probFormat_outputTxt(train_multiS_output_file,train_serial,pos_interval_all,mask_image,classType);
multiSlices_probFormat_outputTxt(val_multiS_output_file,val_serial,pos_interval_all,mask_image,classType);
multiSlices_probFormat_outputTxt(test_multiS_output_file,test_serial,pos_interval_all,mask_image,classType);
multiSlices_probFormat_outputTxt(all_test_multiS_output_file,all_test_serial,pos_interval_all,mask_image,classType);








end

function multiSlices_probFormat_outputTxt(outputFileName,serial_infor_struct,pos_interval_all,mask_image,classType)
	num_serial =length(serial_infor_struct);
	if strcmp(classType,'multi')
		[f_p,f_n,f_e]=fileparts(outputFileName);
		outputFileName=[f_p filesep f_n '_multi_class' f_e];
	end
	fileID = fopen(outputFileName,'w');

	for i=1:num_serial
		section_info=serial_infor_struct(i);
		index_closest=getIndex_Pos_Closed2interaval(section_info,pos_interval_all);
		imNo=section_info.imagesNo(index_closest);
		labels=section_info.lable;
		level_5_pattern_labels=labels{7}(:,1);
		disp(classType)
		if strcmp(classType,'binary')
			txtLabelString =generate_label_txtlineString( level_5_pattern_labels);
		else strcmp(classType,'multi')
			txtLabelString =generate_label_withMultiClass_txtlineString( level_5_pattern_labels);
		end
		serialNo=section_info.serialNo;
		fprintf(fileID,'%s{\n','ISH_section_info');
		fprintf(fileID,'sectionID:%s\n',serialNo);
		
		if mask_image
		  posfix='_msked.jpg';
		else
		  posfix ='.jpg';
		end
		for j=1:length(imNo)
		    
			imgfile=translatePath([section_info.serialDir filesep num2str(imNo(j)) posfix]);
			fprintf(fileID,'%s{\n','image');
			fprintf(fileID,'fileName:"%s"\n}\n',imgfile);
		end
		fprintf(fileID,'labels:"%s"\n}\n',txtLabelString);
	end
    fclose(fileID);

end



function outputTxt(textFile,serial_infor_struct,pos_interval_all,mask_image,classType)
 num_serial =length(serial_infor_struct);
 if strcmp(classType,'multi')
		[f_p,f_n,f_e]=fileparts(textFile);
		textFile=[f_p filesep f_n '_multi_class' f_e];
 end
 fileID = fopen(textFile,'w');
 %formatingString ='%s';
 %imgfiles={};
 %imcount =0;
 if nargin<4
   mask_image=false;
 end
 for i=1:num_serial
     
     section_info=serial_infor_struct(i);
	
     index_closest=getIndex_Pos_Closed2interaval(section_info,pos_interval_all);
	 
	 
     imNo=section_info.imagesNo(index_closest);
     labels=section_info.lable;
     level_5_pattern_labels=labels{7}(:,1); 
	 if strcmp(classType,'binary')
		txtLineString =generate_label_txtlineString( level_5_pattern_labels);
	 elseif strcmp(classType,'multi')
		txtLineString =generate_label_withMultiClass_txtlineString( level_5_pattern_labels);
	 end
     for j=1:length(imNo)
	    if mask_image
           imgfile=translatePath([section_info.serialDir filesep num2str(imNo(j)) '_msked.jpg']);
		else
		   imgfile=translatePath([section_info.serialDir filesep num2str(imNo(j)) '.jpg']);
		end
         outputLine =[imgfile ' ' txtLineString];
         fprintf(fileID,'%s\n',outputLine);
         %imgfiles(imcount)=imgfile;
     end
 end
 fclose(fileID);
end

function txtLineString =generate_label_withMultiClass_txtlineString(labels)
lbs=labels-1;
idx=lbs<0;
lbs(idx)=0;
numformat='%d';
for i=2:length(labels)
    numformat=[numformat ' %d'  ];
end
numString=sprintf(numformat,lbs);
%txtLineString=[filename ' ' numString];
txtLineString=  numString;  
end









function txtLineString =generate_label_txtlineString(labels)
neg_idx=labels==1;  % make all other label(0 2 3 4) as 1 denoated as detected
labels(neg_idx)=-1;
labels(labels>1)=1;
%unannotated =labels==0; %find out cann't annoted label (0)
%labels(unannotated)=-1;  % make cann't annoted label same as untedteded which is 0;
%% -------------------------------------------------------------------
% KEEPING unannoated labels as 0, so that CNN multilabel training will
% ignor this label as it will not be counted in the loss fucntion.

%% ----------------------------------------------------------------
numformat='%d';
for i=2:length(labels)
    numformat=[numformat ' %d'  ];
end
numString=sprintf(numformat,labels);
%txtLineString=[filename ' ' numString];
txtLineString=  numString;  
end

function index_closest=getIndex_Pos_Closed2interaval(section_info_struct,pos_interval_all)
pos=section_info_struct.pos;
dist_p=zeros(1,length(pos));
index_closest=zeros(1,length(pos_interval_all));
for i=1:length(pos_interval_all)
    cur_interv=pos_interval_all(i);
    for j=1:length(pos)
        dist_p(j)=abs(cur_interv-pos(j));
    end
    [~,idx]=min(dist_p);
    index_closest(i)=idx;
end
end


