function run_svm_train(inf_exchange_file_name)
tempDir=translatePath('Z:\ABA\automatedGeneLabelingProject\manannot\temp'); 
addpath('/home/tzeng/ABA/automatedGeneLabelingProject/code');
MPI_info_exchange_file =[inf_exchange_file_name '.mat'];
%MPI_info_exchange_file =[tempDir filesep MPI_info_exchange_file];
load(MPI_info_exchange_file);



%if (fineTune)
%   disp('do svm on NoneTune')
[path,model_name,ext]=fileparts(model_file);
if strcmp(featureType,'singleImage') || strcmp(featureType,'pairedImage')
	if strcmp(partition_mode,'SagiPartition')
		CAFFE_matrix_file=['Caffe_7'  partition_mode '_' lower(feature_sample_mode) '_Matrix_' Age '_L_' num2str(Layer) '_' model_name];
	else
		CAFFE_matrix_file=['Caffe_'  partition_mode '_' lower(feature_sample_mode) '_Matrix_' Age '_L_' num2str(Layer) '_' model_name];
	end
elseif strcmp(featureType,'multiImage')
       CAFFE_matrix_file=['Caffe_Matrix_' Age '_L_' num2str(Layer) '_' model_name];
end
%else
%   disp('do svm on FineTune ')
%	CAFFE_matrix_file=['Caffe_' partition_mode '_' feature_sample_mode 'FineTune''_Matrix_'];
%end
%train_CAFFE_linear_on_All_MPI(Age,Layer,CAFFE_matrix_file,feature_sample_mode,fineTune);

train_CAFFE_linearSVM_All(Age,CAFFE_matrix_file);
end