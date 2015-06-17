function generate_txtFile_for_Caffe_input( age, numOfImagesForEachSerial,train_ratio,val_ratio)
addpath('/home/tzeng/ABA/automatedGeneLabelingProject/code');
[serial_section_info ,max_sagital_sectionPos]= getSerial_Info_ByAge(age);
num_serail=length(serial_section_info);


%'train_validataion_test_set_for_caffeCNN_level' num2str(ontology_level) '_' age '.mat'

data_dir=translatePath('Z:\ABA\autoGenelable_multi_lables_proj\data');
filename =['train_validataion_test_set_for_caffeCNN_' age '.mat'];
filename=[data_dir filesep filename];
pos_interval=max_sagital_sectionPos/(numOfImagesForEachSerial-1);
pos_interval_all=0:pos_interval:max_sagital_sectionPos;
pos_interval_all(1)=1;

if exist(filename,'file')==2
    load(filename);
else
    perm_serial_idx=randperm(num_serail);
    t_n = round(train_ratio*num_serail);
    v_n = round(val_ratio*num_serail);
    train_serial = serial_section_info(perm_serial_idx(1:t_n));
    val_serial   = serial_section_info(perm_serial_idx(t_n+1:t_n+v_n));
    test_serial  = serial_section_info(perm_serial_idx(t_n+v_n+1:end));
    save(filename,'train_serial','val_serial','test_serial');
   % save(filename,'train_serial','val_serial','test_serial');
end
train_output_file =     [data_dir filesep 'train.txt'];
val_output_file   =     [data_dir filesep 'val.txt'];
test_output_file  =     [data_dir filesep 'test.txt'];

train_multiS_output_file =     [data_dir filesep 'train_slice_img.txt'];
val_multiS_output_file   =     [data_dir filesep 'val_slice_img.txt'];
test_multiS_output_file  =     [data_dir filesep 'test_slice_img.txt'];

outputTxt(train_output_file,train_serial,pos_interval_all);
outputTxt(val_output_file,val_serial,pos_interval_all);
outputTxt(test_output_file,test_serial,pos_interval_all);

multiSlices_probFormat_outputTxt(train_multiS_output_file,train_serial,pos_interval_all);
multiSlices_probFormat_outputTxt(val_multiS_output_file,val_serial,pos_interval_all);
multiSlices_probFormat_outputTxt(test_multiS_output_file,test_serial,pos_interval_all);


end
function multiSlices_probFormat_outputTxt(outputFileName,serial_infor_struct,pos_interval_all)
	num_serial =length(serial_infor_struct);
	fileID = fopen(outputFileName,'w');

	for i=1:num_serial
		section_info=serial_infor_struct(i);
		index_closest=getIndex_Pos_Closed2interaval(section_info,pos_interval_all);
		imNo=section_info.imagesNo(index_closest);
		labels=section_info.lable;
		level_5_pattern_labels=labels{7}(:,1);
		txtLabelString =generate_label_txtlineString( level_5_pattern_labels);
		serialNo=section_info.serialNo;
		fprintf(fileID,'%s{\n','ISH_section_info');
		fprintf(fileID,'sectionID:%s\n',serialNo);
		for j=1:length(imNo)
			imgfile=translatePath([section_info.serialDir filesep num2str(imNo(j)) '.jpg']);
			fprintf(fileID,'%s{\n','image');
			fprintf(fileID,'fileName:"%s"\n}\n',imgfile);
		end
		fprintf(fileID,'labels:"%s"\n}\n',txtLabelString);
	end
    fclose(fileID);

end


function outputTxt(textFile,serial_infor_struct,pos_interval_all)
 num_serial =length(serial_infor_struct);
 fileID = fopen(textFile,'w');
 %formatingString ='%s';
 %imgfiles={};
 %imcount =0;
 for i=1:num_serial
     
     section_info=serial_infor_struct(i);
     index_closest=getIndex_Pos_Closed2interaval(section_info,pos_interval_all);
     imNo=section_info.imagesNo(index_closest);
     labels=section_info.lable;
     level_5_pattern_labels=labels{7}(:,1);
     txtLabelString =generate_label_txtlineString( level_5_pattern_labels);
     for j=1:length(imNo)
         %imgfile=translatePath([section_info.serialDir filesep num2str(imNo(j)) '_msked.jpg']);
		 imgfile=translatePath([section_info.serialDir filesep num2str(imNo(j)) '.jpg']);
         outputLine =[imgfile ' ' txtLabelString];
         fprintf(fileID,'%s\n',outputLine);
         %imgfiles(imcount)=imgfile;
     end
 end
 fclose(fileID);
end

function txtLineString =generate_label_txtlineString(labels)
neg_idx=labels==1;  % make all other label(0 2 3 4) as 1 denoated as detected
labels(neg_idx)=-1;
labels(labels>1)=1;
unannotated =labels==0; %find out cann't annoted label (0)
labels(unannotated)=-1;  % make cann't annoted label same as untedteded which is 0;
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