%% This is vgg_16 layer info
% 0('data', (10, 3, 224, 224))
% 1('conv1_1', (10, 64, 224, 224))
% 2('conv1_2', (10, 64, 224, 224))
% 3('pool1', (10, 64, 112, 112))
% 4('conv2_1', (10, 128, 112, 112))
% 5('conv2_2', (10, 128, 112, 112))
% 6('pool2', (10, 128, 56, 56))
% 7('conv3_1', (10, 256, 56, 56))
% 8('conv3_2', (10, 256, 56, 56))
% 9('conv3_3', (10, 256, 56, 56))
% 10('pool3', (10, 256, 28, 28))
% 11('conv4_1', (10, 512, 28, 28))
% 12('conv4_2', (10, 512, 28, 28))
% 13('conv4_3', (10, 512, 28, 28))
% 14('pool4', (10, 512, 14, 14))
% 15('conv5_1', (10, 512, 14, 14))
% 16('conv5_2', (10, 512, 14, 14))
% 17('conv5_3', (10, 512, 14, 14))
% 18('pool5', (10, 512, 7, 7))
% 19('fc6', (10, 4096, 1, 1))
% 20('fc7', (10, 4096, 1, 1))
% 21('fc8', (10, 1000, 1, 1))
% 22('prob', (10, 1000, 1, 1))

function set_run_all_param(ImageType,model_def_file,model_file,Layer,Age,inf_exchange_file_name,deviceID,featureType)
	tempDir=translatePath('Z:\ABA\automatedGeneLabelingProject\manannot\temp');        
	partition_mode      =	'whole';
	feature_sample_mode =	'max';
	if nargin <5
		Age                 =	'E11.5'
	end
	if nargin <4
		Layer               =	'conv5_3';
	end
	addpath('/home/tzeng/ABA/automatedGeneLabelingProject/code');
	MPI_info_exchange_file =[inf_exchange_file_name '.mat'];
	%MPI_info_exchange_file =[tempDir filesep MPI_info_exchange_file];
	%deviceID =1;


	save(MPI_info_exchange_file, 'Age','Layer','feature_sample_mode','partition_mode','deviceID','model_def_file','model_file','ImageType','featureType');
end