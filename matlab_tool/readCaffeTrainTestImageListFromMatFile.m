function [trainData,valData,testData,remainData] = readCaffeTrainTestImageListFromMatFile(age,ontology_level,ImageType)
	addpath(translatePath('Z:\ABA\automatedGeneLabelingProject\code'));
	numOfsectionForEachSerial =10;
	data_dir=translatePath('Z:\ABA\autoGenelable_multi_lables_proj\data');
	data_dir_cur=translatePath('Z:\ABA\autoGenelable_multi_lables_proj\data');
	filename =['train_validataion_test_set_for_caffeCNN_level' num2str(ontology_level) '_' age '.mat'];
	filename=[data_dir_cur filesep filename];
	load(filename);

	max_sagital_sectionPos=getMaxSagitalSectionPos(age);
	
	
	if strcmp(ImageType,'ISH')
		use_mask_image =false;
	elseif strcmp(ImageType,'MSK')
		use_mask_image    =true;
	end
	
	
	
	testData=getSerialInfoForFeatureExtraction(test_serial,numOfsectionForEachSerial,max_sagital_sectionPos,use_mask_image);
	trainData=getSerialInfoForFeatureExtraction(train_serial,numOfsectionForEachSerial,max_sagital_sectionPos,use_mask_image);
	valData=getSerialInfoForFeatureExtraction(val_serial,numOfsectionForEachSerial,max_sagital_sectionPos,use_mask_image);
	remainData=getSerialInfoForFeatureExtraction(remain_serial,numOfsectionForEachSerial,max_sagital_sectionPos,use_mask_image);

end


function serial_Data=getSerialInfoForFeatureExtraction(serial_info_struct,numOfsectionForEachSerial,max_sagital_sectionPos,mask_image)
		pos_interval=max_sagital_sectionPos/(numOfsectionForEachSerial-1);
		pos_interval_all=0:pos_interval:max_sagital_sectionPos;
		pos_interval_all(1)=1;
		for i=1:length(serial_info_struct)
			section_info=serial_info_struct(i);
			index_closest=getIndex_Pos_Closed2interaval(section_info,pos_interval_all);
			imNo=section_info.imagesNo(index_closest);
			labels=section_info.lable;
			level_5_pattern_labels=labels{7}(:,1);
			%txtLabelString =generate_label_txtlineString( level_5_pattern_labels);
			serial_Data(i).serialNo=section_info.serialNo;
			serial_Data(i).labels   =level_5_pattern_labels;
		
			if mask_image
			  posfix='_msked.jpg';
			else
			  posfix ='.jpg';
			end
			for j=1:length(imNo)
				imgfile=translatePath([section_info.serialDir filesep num2str(imNo(j)) posfix]);
				serial_Data(i).imageFile{j}=imgfile;
			end
		end
end