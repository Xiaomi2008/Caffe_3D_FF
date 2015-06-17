function  featureOut=caffe_feature_extraction_sliceImages(Age,net_layer_name, deviceID,model_def_file,model_file,ImageType)
	addpath(translatePath('Z:\ABA\autoGenelable_multi_lables_proj\code\caffe\matlab\caffe'));
	if nargin==0
		Age='E11.5';
	end
	
	[trainData,valData,testData ] = readCaffeTrainTestImageListFromTextFile(Age,ImageType);
	data_dir_cur=translatePath('Z:\ABA\autoGenelable_multi_lables_proj\data');
    filename =['train_validataion_test_set_for_caffeCNN_level' num2str(ontology_level) '_' age '.mat'];
    filename=[data_dir_cur filesep filename];
    load(filename);
    
    
    
    
    
	if nargin <2
		net_layer_name ='conv5_1';
		deviceID  =2;
	end
	%
	[path,model_name,ext]=fileparts(model_file);
	disp(['extract feature name from model file ' model_file '  is ' model_name])
	% if ~fineTune
	% featureMatFile_suffix =['_VGG_16_L_' net_layer_name '_caffeFeature'];
	% else
	% featureMatFile_suffix =['_VGG_16_L_' net_layer_name '_FineTune_caffeFeature'];
	% end
	featureMatFile_suffix =[model_name '_' net_layer_name];
	
	
	%% init caffe net
	
	% if ~fineTune
		% model_def_file='/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/vgg_16/deploy.prototxt';
		% model_file = '/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/vgg_16/VGG_16.caffemodel';
	% else
	   	% model_def_file='/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/vgg_16_fine_tune_gen/deploy.prototxt';
		% model_file = '/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/finetune_vgg_16_brain_onotology_3color_mean_noCrop_last_L_3-6_L0.001_step2500_iter_12000';
	% end
	
	
	
	net = CaffeNet.instance(model_def_file,model_file);
	net.set_mode_gpu()
	net.set_device(deviceID);
	net.set_phase_test;

	disp('extracting train set features ...');
	extract_feature_set(net,trainData,net_layer_name,featureMatFile_suffix);
	disp('extracting validation set features ...');
	extract_feature_set(net,valData,net_layer_name,featureMatFile_suffix);
	disp('extracting test set features ...');
	featureOut=extract_feature_set(net,testData,net_layer_name,featureMatFile_suffix);
	%featureOut =   extract_feature(net,imageFile,net_layer_name);


end

function featureOut=extract_feature_set(net,dataStruct,layer_name,saveFileSuffix)
%masked_ext_length =   length('_msked');
%for i=1:length(dataStruct.imageFile)
for i=1:length(dataStruct.imageFile)
     imagefile =dataStruct.imageFile{i};
     [pathstr,name,ext]=fileparts(imagefile);    
     %outMatFile=[pathstr filesep name(1:end-masked_ext_length) saveFileSuffix '_.mat']; 
	 outMatFile=[pathstr filesep name '_' saveFileSuffix '_.mat']; 
    
     % if exist(outMatFile,'file') ==2
         % disp(['feature exsit :  ' outMatFile]);
         % continue;
     % end
	 
	 featureOut =extract_feature(net,imagefile,layer_name);
	 [w,h, n]=size(featureOut);
	 mapsize(1)=w;
	 mapsize(2)=h;
	 mapsn     =n;
	 
	 
	 disp(['Saving feature to :  ' outMatFile]);
	 save(outMatFile,'featureOut','mapsize','mapsn');
end

end
