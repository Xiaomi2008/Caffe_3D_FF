function run_build_AllsectionFeature2Mat(inf_exchange_file_name)
tempDir=translatePath('Z:\ABA\automatedGeneLabelingProject\manannot\temp'); 
addpath('/home/tzeng/ABA/automatedGeneLabelingProject/code');
MPI_info_exchange_file =[inf_exchange_file_name '.mat'];
%MPI_info_exchange_file =[tempDir filesep MPI_info_exchange_file];
load(MPI_info_exchange_file);
[p,model_name,ext]=fileparts(model_file);
build_Caffe_Matrix(model_name,Age,Layer,partition_mode, feature_sample_mode,featureType) ; 
end