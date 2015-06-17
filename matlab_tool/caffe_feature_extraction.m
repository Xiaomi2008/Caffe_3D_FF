function  caffe_feature_extraction(Age,net_layer_name, deviceID,model_def_file,model_file,ImageType,featureType)
	addpath(translatePath('Z:\ABA\autoGenelable_multi_lables_proj\code\caffe\matlab\caffe'));
	if nargin==0
		Age='E11.5';
	end
	
	
	

	if nargin <2
		net_layer_name ='conv5_1';
		deviceID  =2;
	end
	%
	
	[path,model_name,ext]=fileparts(model_file);
	disp(['extract feature name from model file ' model_file])
	disp(['and model define model  is: ' model_def_file])
	
	featureMatFile_suffix =[model_name '_' net_layer_name];
	
	allsectionFeatureMatFile_suffix =[net_layer_name '_' model_name ];
	
	
	%% init caffe net
	
	% if ~fineTune
		% model_def_file='/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/vgg_16/deploy.prototxt';
		% model_file = '/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/vgg_16/VGG_16.caffemodel';
	% else
	   	% model_def_file='/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/vgg_16_fine_tune_gen/deploy.prototxt';
		% model_file = '/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/finetune_vgg_16_brain_onotology_3color_mean_noCrop_last_L_3-6_L0.001_step2500_iter_12000';
	% end
	
	
	
	net = CaffeNet.instance(model_def_file,model_file);
	disp('caffe net initialization finished')
	disp(['extract feature name from model file ' model_file])
	disp(['and model define model  is: ' model_def_file])
	net.set_mode_gpu()
	net.set_device(deviceID);
	net.set_phase_test;
    
	if nargin <7
	  featureType = 'singleImage';
	end
	
	if strcmp(featureType,'multiImage')
	     ontology_level =7;
	     [trainData,valData,testData,remainData ] =  readCaffeTrainTestImageListFromMatFile(Age,ontology_level,ImageType);
		disp('extracting train set features ...');
		extract_feature_set_fromSliceImages(net,trainData,net_layer_name,allsectionFeatureMatFile_suffix);
		disp('extracting test set features ...');
		extract_feature_set_fromSliceImages(net,testData,net_layer_name,allsectionFeatureMatFile_suffix);
		disp('extracting validation set features ...');
		extract_feature_set_fromSliceImages(net,valData,net_layer_name,allsectionFeatureMatFile_suffix);
		disp('extracting remaining set features ...');
		extract_feature_set_fromSliceImages(net,remainData,net_layer_name,allsectionFeatureMatFile_suffix);
	elseif strcmp(featureType, 'pairedImage')
		ontology_level =7;
	     [trainData,valData,testData,remainData ] =  readCaffeTrainTestImageListFromMatFile(Age,ontology_level,ImageType);
		disp('extracting train set features ...');
		extract_feature_set_fromPairedImages(net,trainData,net_layer_name,featureMatFile_suffix);
		disp('extracting test set features ...');
		extract_feature_set_fromPairedImages(net,testData,net_layer_name,featureMatFile_suffix);
		disp('extracting validation set features ...');
		extract_feature_set_fromPairedImages(net,valData,net_layer_name,featureMatFile_suffix);
		disp('extracting remaining set features ...');
		extract_feature_set_fromPairedImages(net,remainData,net_layer_name,featureMatFile_suffix);
		
	
	elseif strcmp(featureType,'singleImage')
	   	[trainData,valData,testData ] = readCaffeTrainTestImageListFromTextFile(Age,ImageType);
		disp('extracting train set features ...');
		extract_feature_set(net,trainData,net_layer_name,featureMatFile_suffix);
		disp('extracting validation set features ...');
		extract_feature_set(net,valData,net_layer_name,featureMatFile_suffix);
		disp('extracting test set features ...');
		extract_feature_set(net,testData,net_layer_name,featureMatFile_suffix);
	end
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


function featureOut=extract_feature_set_fromSliceImages(net,dataStruct,layer_name,saveFileSuffix)
%masked_ext_length =   length('_msked');
%for i=1:length(dataStruct.imageFile)
len=length(dataStruct);
	for i=1:len
	    [pathstr,name,ext]=fileparts(dataStruct(i).imageFile{1});
		%'All_Section'  '_L_' num2str(layer_name) '_CaffeSlice_' model_name '.mat'
		
		
	    outMatFile=[pathstr filesep 'All_Section_L' saveFileSuffix '_.mat']; 
		for j=1:length(dataStruct(i).imageFile)
		 imagefile{j} =dataStruct(i).imageFile{j};
		end 
		featureOut =extract_feature(net,imagefile,layer_name);
		ALL_SECTION_OFF=featureOut(:); 
		% [w,h, n]=size(featureOut);
		% mapsize(1)=w;
		% mapsize(2)=h;
		% mapsn     =n;
		imageFiles=dataStruct(i).imageFile;
		 disp(['Saving feature to :  ' outMatFile ' # serial =' num2str(i) ' : total =' num2str(len) ]);
		 save(outMatFile,'ALL_SECTION_OFF','imageFiles');
	end

end


function featureOut=extract_feature_set_fromPairedImages(net,dataStruct,layer_name,saveFileSuffix)
%masked_ext_length =   length('_msked');
%for i=1:length(dataStruct.imageFile)
len=length(dataStruct);
	for i=1:len
	
	       
		 %outMatFile=[pathstr filesep name(1:end-masked_ext_length) saveFileSuffix '_.mat']; 
		% outMatFile=[pathstr filesep name '_' saveFileSuffix '_.mat']; 
		 
		 
	    %[pathstr,name,ext]=fileparts(dataStruct(i).imageFile{1});
		%'All_Section'  '_L_' num2str(layer_name) '_CaffeSlice_' model_name '.mat'
		 
		for j=1:2:length(dataStruct(i).imageFile)
		 im_name_1=dataStruct(i).imageFile{j};
		 im_name_2=dataStruct(i).imageFile{j+1};
		 
		 imagefile{1}=im_name_1;
		 imagefile{2}=im_name_2;
		 [pathstr1,name1,ext1]=fileparts(im_name_1); 
		 [pathstr2,name2,ext2]=fileparts(im_name_2); 
		 %imagefile{j} =dataStruct(i).imageFile{j};
		 
		 outMatFile=[pathstr1 filesep name1 '_' saveFileSuffix '_.mat'];
		 featureOut =extract_feature(net,imagefile,layer_name); 
		 [w,h, n]=size(featureOut);
		 mapsize(1)=w;
		 mapsize(2)=h;
		 mapsn     =n;
		
		
		disp(['Saving feature to :  ' outMatFile ' # serial =' num2str(i) '+' ' : total =' num2str(len) ]);
		%disp(['Saving feature to :  ' outMatFile]);
		save(outMatFile,'featureOut','mapsize','mapsn');
		end 
	end

end




