function trainPredict_on_caffe_featuer_parfor(trainMatStruct,testMatStruct,savefile)
	 if ~ispc
		addpath('/home/tzeng/ABA/automatedGeneLabelingProject/code');
	else 
		addpath('Z:\ABA\automatedGeneLabelingProject\code');
	end
	temp_dir =translatePath('Z:\ABA\autoGenelable_multi_lables_proj\temp');
	data_dir =translatePath('Z:\ABA\autoGenelable_multi_lables_proj\data');
	train=norm_bow_matrix(trainMatStruct.feat);
	test=norm_bow_matrix(testMatStruct.feat);
	disp(['saving result file is  '  savefile])
	num_labels=	 size(trainMatStruct.label,2);
	matlabpool close force local
	matlabpool 8
	parfor i=1:num_labels
	%for i=1:num_labels
		tr_lb=trainMatStruct.label(:,i);
		te_lb=testMatStruct.label(:,i);
	    [out1, out2, out3, out4]= LinearBinSVM(train, tr_lb, test,te_lb, 1);
                     %[auc_value(i-startR+1),  accu(:,i-startR+1), predict_LBs{i-startR+1}]
		auc(i)              =out1;
		
		accu(:,i)                 =out2;
		predict_LBs{i}            =out3;
		predict_probability{i}    =out4;
		true_LBs{i}               = te_lb;
	end
	%matlabpool close;
	%auc=1;
	save(savefile,'auc','accu','predict_LBs','true_LBs');
	%save(savefile,'auc');
end