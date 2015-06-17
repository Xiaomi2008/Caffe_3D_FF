function run_extract_image_feature(inf_exchange_file_name)
	tempDir=translatePath('Z:\ABA\automatedGeneLabelingProject\manannot\temp'); 
	addpath('/home/tzeng/ABA/automatedGeneLabelingProject/code');
	MPI_info_exchange_file =[inf_exchange_file_name '.mat'];
	%MPI_info_exchange_file =[tempDir filesep MPI_info_exchange_file];
	%MPI_info_exchange_file ='MPI_age_layer_info.mat';
	%MPI_info_exchange_file =[tempDir filesep MPI_info_exchange_file];
	load(MPI_info_exchange_file);
	caffe_feature_extraction(Age,Layer, deviceID,model_def_file,model_file,ImageType,featureType);
end