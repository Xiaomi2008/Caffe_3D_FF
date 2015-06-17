tempDir=translatePath('Z:\ABA\automatedGeneLabelingProject\manannot\temp');        
partition_mode      =	'whole';
feature_sample_mode =	'max';
Age                 =	'E11.5'
Layer               =	'conv5_1';

addpath('/home/tzeng/ABA/automatedGeneLabelingProject/code');
MPI_info_exchange_file ='MPI_age_layer_info.mat';
MPI_info_exchange_file =[tempDir filesep MPI_info_exchange_file];
deviceID =3;
fineTune=false;
featureOut=caffe_feature_extraction(Age,Layer, deviceID,fineTune);
% save(MPI_info_exchange_file, 'Age','Layer','feature_sample_mode','finetune');
% nump=32;

% system(['mpirun -np ' num2str(nump) ' -bycore -machinefile hostname octave run_build_caffe_sectionFeature_octave_MPI.m']);
% delete MPI_info_exchange_file;

%build_Caffe_Matrix(fineTune,Age,Layer,partition_mode, feature_sample_mode) ; 

if (fineTune)
	CAFFE_matrix_file=['Caffe_' partition_mode '_' feature_sample_mode '_Matrix_'];
else
	CAFFE_matrix_file=['Caffe_' partition_mode '_' feature_sample_mode 'FineTune''_Matrix_'];
end
train_CAFFE_linear_on_All_MPI(Age,Layer,CAFFE_matrix_file,feature_sample_mode,fineTune);