clear all
if ispc
data_dir='z:\caffe_3d\data';
else
data_dir ='/home/tzeng/caffe_3d/data';
end
img_dir =[data_dir filesep 'trecvid'  filesep 'Dev08BMP60x40'];
train_outputFileName = [data_dir filesep 'action_data_set_train.txt'];
test_outputFileName = [data_dir filesep 'action_data_set_test.txt'];
root_dir_struct =dir(img_dir);
frame_dir = {'-8' '-6' '-4' '-2' '0' '2' '4' '6' '8'};
num_cam_dir_struct =dir([img_dir filesep frame_dir{1} ]);
cam_dir_names ={num_cam_dir_struct(3:end).name};
count =1;
for i=1:length(cam_dir_names)
	cam_pic_file_struct =dir([img_dir filesep frame_dir{1} filesep cam_dir_names{i}]);
	cam_dir_file_names={cam_pic_file_struct(3:end).name};
    disp(['working on ' num2str(i) 'th out of ' num2str(length(cam_dir_names))])
	for j=1:length(cam_dir_file_names) 
        filenm  =cam_dir_file_names{j};
        if strcmp(filenm(end-3:end),'.bmp')
            for k=1:length(frame_dir)
                    file_path_name = [img_dir filesep frame_dir{k} filesep cam_dir_names{i} filesep filenm];
                    file_all(count).filenams{k}=file_path_name;
            end
           
        
            if strcmp('CellToEar',filenm(1:9))
                    label = 1;
            end
            if strcmp('Negative',filenm(1:8))
                    label = 0;
            end
            if strcmp('Pointing',filenm(1:8))
                    label = 2;
            end
            if strcmp('ObjectPut',filenm(1:9))
                    label = 3;
            end
            file_all(count).label=label;
             count=count+1;
        end
        
                
	end
end

num_sample=length(file_all);
r_idx=randperm(num_sample);
file_all=file_all(r_idx);
part_pos=floor(length(r_idx)/3 *2)
train_file_all=file_all(1:part_pos);
test_file_all=file_all(part_pos+1:end);

labels={file_all.label};
labels=cell2mat(labels);

idx_c0=labels==0;
idx_c1=labels==1;
idx_c2=labels==2;
idx_c3=labels==3;

c0=file_all(idx_c0);
c0=c0(1:3:12000*3);
c1=file_all(idx_c1);
c2=file_all(idx_c2);
c3=file_all(idx_c3);

balanced_file=[c0, c1, c2 c3];
num_sample=length(balanced_file);
r_idx=randperm(num_sample);
balanced_file=balanced_file(r_idx);
part_pos=floor(length(r_idx)/3 *2)
train_file_all=balanced_file(1:part_pos);
test_file_all=balanced_file(part_pos+1:end);


disp('writting train set ...')
num_samples=length(train_file_all);
fileID = fopen(train_outputFileName,'w');
for i=1:num_samples
        fprintf(fileID,'%s{\n','ISH_section_info');
        fprintf(fileID,'sectionID:%s\n',num2str(i));
        for j=1:length(frame_dir)
			imgfile= train_file_all(i).filenams(j);
			fprintf(fileID,'%s{\n','image');
			fprintf(fileID,'fileName:"%s"\n}\n',imgfile{1});
		end
		fprintf(fileID,'labels:"%s"\n}\n',num2str(train_file_all(i).label));
    
end

  fclose(fileID);
  
  
  disp('writting test set...')
  num_samples=length(test_file_all);
fileID = fopen(test_outputFileName,'w');
for i=1:num_samples
        fprintf(fileID,'%s{\n','ISH_section_info');
        fprintf(fileID,'sectionID:%s\n',num2str(i));
        for j=1:length(frame_dir)
			imgfile= test_file_all(i).filenams(j);
			fprintf(fileID,'%s{\n','image');
			fprintf(fileID,'fileName:"%s"\n}\n',imgfile{1});
		end
		fprintf(fileID,'labels:"%s"\n}\n',num2str(test_file_all(i).label));
    
end

  fclose(fileID);

% 
% file_structs = [img_dir filesep frame_dir{1} filesep cam_dir_names{1}]
% 
% fileID = fopen(outputFileName,'w');
% 	for i=1:num_serial
% 		section_info=serial_infor_struct(i);
% 		index_closest=getIndex_Pos_Closed2interaval(section_info,pos_interval_all);
% 		imNo=length(frame_dir);
% 		labels=section_info.lable;
% 		level_5_pattern_labels=labels{7}(:,1);
% 		txtLabelString =generate_label_txtlineString( level_5_pattern_labels);
% 		serialNo=section_info.serialNo;
% 		fprintf(fileID,'%s{\n','ISH_section_info');
% 		fprintf(fileID,'sectionID:%s\n',serialNo);
% 		for j=1:length(imNo)
% 			imgfile=translatePath([section_info.serialDir filesep num2str(imNo(j)) '.jpg']);
% 			fprintf(fileID,'%s{\n','image');
% 			fprintf(fileID,'fileName:"%s"\n}\n',imgfile);
% 		end
% 		fprintf(fileID,'labels:"%s"\n}\n',txtLabelString);
% 	end
%     fclose(fileID);