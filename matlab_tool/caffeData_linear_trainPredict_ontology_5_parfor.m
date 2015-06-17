 function caffeData_linear_trainPredict_ontology_5_parfor(inputMat)
	 if ~ispc
		addpath('/home/tzeng/ABA/automatedGeneLabelingProject/code');
	else 
		addpath('Z:\ABA\automatedGeneLabelingProject\code');
	end
	temp_dir =translatePath('Z:\ABA\autoGenelable_multi_lables_proj\temp');
	data_dir =translatePath('Z:\ABA\autoGenelable_multi_lables_proj\data');
	Matrixfile ='tempTrainFeatures';
	
	
	
	source_fileP1     =fullfile(temp_dir,[Matrixfile 'MatP1.mat']);
	source_fileP2     =fullfile(temp_dir,[Matrixfile 'MatP2.mat']);
	source_fileP3     =fullfile(temp_dir,[Matrixfile 'MatP3.mat']);
	source_fileP4     =fullfile(temp_dir,[Matrixfile 'MatP4.mat']);
	labefile          =fullfile(temp_dir, 'tempTrainFeatures_labels.mat');
	save_file         =fullfile(temp_dir, 'tempTrainResult.mat');
	
	load(labefile);
	if nargin<1
	matP1=load(source_fileP1);
	matP2=load(source_fileP2);
	matP3=load(source_fileP3);
	matP4=load(source_fileP4);
    % get labels from reading the label file 
    
	
	featureMat=[matP1.Mat' matP2.Mat' matP3.Mat' matP4.Mat']';
	clear matP1.Mat;
	clear matP2.Mat;
	clear matP3.Mat;
	clear matP4.Mat;
	featureMat=sparse(featureMat);
	else
	featureMat=sparse(inputMat);
	clear inputMat;
	end
	
	
	
	auc_value =[];
	accu      =[];
     all_level_labs=[];
     ontology_mat  =[];
     mod_mat       =[];
     structName    ={};
	 [train_IDX,test_IDX]=find_TrainTest_IDX_from_predefined(dataSetsID,'E11.5');
    for ontology =7:7
        ontology_level = ontology;
        cur_struct_names =getSpecifiedStructName(structureName,ontology_level);
        mod='pattern';
        level_labs_pattern=getSpecifiedLabels(labels,mod,ontology_level);
       
        %size(level_labs_pattern)
        all_level_labs	= [all_level_labs level_labs_pattern];
        ontology_mat 	= [ontology_mat ones(1,size(level_labs_pattern,2))*ontology_level];
        mod_mat      	= [mod_mat ones(1,size(level_labs_pattern,2))*1];   
        structName      = [structName cur_struct_names'];
		
    end
	
	disp(['size of  all_level_labs is ' num2str(size(all_level_labs,1)) ' X '  num2str(size(all_level_labs,2))]);
	 
	Train				=featureMat(train_IDX, :);
	Test 				=featureMat(test_IDX,:);
	test_dataSetsID		=dataSetsID(test_IDX);
	cur_test_dataSetsID =test_dataSetsID;
	disp(['mod_mat size = ' num2str(length(mod_mat))]);
	if matlabpool('size')>0
		matlabpool close
    end
     matlabpool 8;
     numserial =	 size(all_level_labs,2);
	 parfor i=1:numserial
	    level_labs=all_level_labs(:,i); 
        tr_label  =level_labs(train_IDX,:);
		te_label  =level_labs(test_IDX,:);	
		% disp('tr_label')
		% tr_label
		% disp('te_label')
		% te_label
		
		
		Tr_zIdx=tr_label~=0;
		Train1=Train(Tr_zIdx,:);
		tr_label1=tr_label(Tr_zIdx);
		
		Te_zIdx=te_label~=0;
		Test1=Test(Te_zIdx,:);
		te_label1=te_label(Te_zIdx);
		
		cur_test_dataSetsID1=cur_test_dataSetsID(Te_zIdx);
		
		[out1, out2, out3, out4]= LinearBinSVM(Train1, tr_label1, Test1,te_label1, 1);
                     %[auc_value(i-startR+1),  accu(:,i-startR+1), predict_LBs{i-startR+1}]
		auc_value(i)              =out1;
		
		accu(:,i)                 =out2;
		predict_LBs{i}            =out3;
		predict_probability{i}    =out4;
		
		
		
		te_dataSetsIDs{i} =cur_test_dataSetsID1;
        true_LBs{i}= te_label1;
	 end
	 matlabpool close;
	 test_dataSetsID =te_dataSetsIDs;
	 predict_probs   = predict_probability;
	 save(save_file,'auc_value','accu','ontology_mat','mod_mat','structName','test_dataSetsID','predict_LBs','true_LBs','predict_probs');
end


	