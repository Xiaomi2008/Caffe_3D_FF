%function run_build_caffe_sectionFeature_ovtave_MPI(MPI_info_exchange_file)
addpath('/home/tzeng/ABA/automatedGeneLabelingProject/code')
tempDir =translatePath('/home/tzeng/ABA/autoGenelable_multi_lables_proj/temp');
MPI_info_exchange_file ='MPI_age_layer_info.mat';
MPI_info_exchange_file =[tempDir filesep MPI_info_exchange_file];
load(MPI_info_exchange_file);
[path,model_name,ext]=fileparts(model_file);
%buildSectionCaffeFeatures_by_Age_MPI(fineTune,Age,Layer,feature_sample_mode)
buildSectionCaffeFeatures_by_Age_MPI(ImageType,model_name,Age,Layer,feature_sample_mode)
%end