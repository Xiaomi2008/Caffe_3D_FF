function caffeData_linear_trainPredict_ontology_5_MPI()
addpath('/home/tzeng/ABA/automatedGeneLabelingProject/code');
temp_dir =translatePath('Z:\ABA\autoGenelable_multi_lables_proj\temp');
data_dir =translatePath('Z:\ABA\autoGenelable_multi_lables_proj\data');
ovf_data_dir =translatePath('Z:\ABA\automatedGeneLabelingProject\manannot\data');
Matrixfile ='tempTrainFeatures';
source_fileP1     =fullfile(temp_dir,[Matrixfile 'MatP1.mat']);
source_fileP2     =fullfile(temp_dir,[Matrixfile 'MatP2.mat']);
source_fileP3     =fullfile(temp_dir,[Matrixfile 'MatP3.mat']);
source_fileP4     =fullfile(temp_dir,[Matrixfile 'MatP4.mat']);
labefile          =fullfile(temp_dir, 'tempTrainFeatures_labels.mat');
save_file         =fullfile(temp_dir, 'tempTrainResult.mat');

%tempTestFile    =fullfile(temp_dir,'feature_matrix_3.mat');
%Age='E13.5';
%Train          =   train_matrix;
%Train_Ls       =   train_level_labs;
%Test           =   test_matrix;
%Test_Ls        =   test_level_labs;
%lab_len        =   size(level_labs,2);


if ~exist(source_fileP1,'file')
    disp(['data file '  source_file ' does not exist ---  quit MPI ...'])
    return
end

pkg load mpi;
default_save_options('-7');
MPI_Init();

%% ===========================================
% uncommen this to run MPI in octave
CW = MPI_Comm_Load("WHATEVER");
%%===================================================
%WORLD = MPI_COMM_WORLD;
my_rank = MPI_Comm_rank (CW);
p = MPI_Comm_size (CW);
mytag = 1048;
source = 0;
MPI_SUCCESS = 0;
if (my_rank > 1)
    [featureMat, info1]      = MPI_Recv (0, mytag, CW);
    [all_level_labs, info2]  = MPI_Recv (0, mytag+1, CW);
    [ontology_mat, info3]    = MPI_Recv (0, mytag+2, CW);
    [mod_mat, info4]         = MPI_Recv (0, mytag+3, CW);
	[labels, info5]          = MPI_Recv (0, mytag+9, CW);
    [Age, info6]             = MPI_Recv (0, mytag+10, CW); 
    [structName,info7]       = MPI_Recv (0, mytag+20,CW);
    [dataSetsID, info8]      = MPI_Recv (0, mytag+30,CW);
	%[train_IDX,info9]        = MPI_Recv (0, mytag+40,CW);
	%[test_IDX,info10]   	 = MPI_Recv (0, mytag+50,CW);
    lab_num=size(all_level_labs,2);
	disp(['level size = ' num2str(lab_num)]);
	disp(['rank ' num2str(my_rank) ' received data ...'])
    %disp(['size =' num2str(lab_num)]);
    %disp( ['lab_num = ' num2str(lab_num)]); 
    
	
	
	% range = floor( lab_num /(p-2));
    % stepping = 0:p-2;
    % stepping = stepping * (range);
    % %stepping(end) =  200 + 1;
    % stepping(1) = 1;
    % stepping(end) =  lab_num + 1;
    
	
     range = ceil( lab_num /(p-2));
	 disp(['p =' num2str(p) '  range =' num2str(range)]);
     stepping = 0:p-2;
     stepping = stepping * (range);
    % %stepping(end) =  200 + 1;
     stepping(1) = 1;
     stepping(end) =  lab_num + 1;
	
	
	startR = stepping(my_rank-1);
    lastR = stepping(my_rank+1-1)-1;
    
    if (info1==MPI_SUCCESS && info2==MPI_SUCCESS && info3==MPI_SUCCESS && info4==MPI_SUCCESS &&info5==MPI_SUCCESS &&info6 ==MPI_SUCCESS &&info7 ==MPI_SUCCESS ...
	    && info8==MPI_SUCCESS)
        len=size(all_level_labs,2);
        disp(['all_level_labs  = ' num2str(len)])
       
        %size(Train)
        %size(Train_Ls)
        %size(Test)
        %size(Test_Ls)
		disp(['there are '  num2str(sum(ontology_mat~=1)) ' None 1 ontology'])
		auc_value =[];
		accu      =[];
		previousBLFile ='';
		%balanced_OVF_matrix ='  temp';
		%Train=featureMat(train_IDX, :);
		%Test =featureMat(test_IDX,:);
        %test_dataSetsID=dataSetsID_nocAnnot(test_IDX);
		%test_dataSetsID=dataSetsID(test_IDX);
        for i=startR:lastR
            %disp(['i = ' num2str(i)])
            ontology_level =ontology_mat(i);
            mod = mod_mat(i);
		%	disp(['mod = ' num2str(mod)])
		%	disp(['ontology_level  = ' num2str(ontology_level)])
		    
            switch mod
                case 1
                    post_fix = '_pattern.mat';
                case 2
                    post_fix = '_intensity.mat';
                case 3
                    post_fix = '_density.mat';
            end
            %disp('featuremat size =')
            %size(featureMat)
		    %size(all_level_labs)
			
			balanced_OVF_matrix    =fullfile(ovf_data_dir,['balanced_BOW_matrix_Age_' Age 'OntologyLV_' num2str(ontology_level)  post_fix]);
			if ~ strcmp(previousBLFile,balanced_OVF_matrix )  % this avoid redundantly loading same blanced_idx file
				load(balanced_OVF_matrix); 
				previousBLFile  =  balanced_OVF_matrix;
		    end
			
            %disp(['Ontology = :'  balanced_OVF_matrix]);
			disp(['Ontology = ' num2str(ontology_level) ': Measure = ' post_fix(2:end-4) ': Machine  = ' num2str(my_rank) ': Label N = '  num2str(i) 'th label']);
            
			
			%idx=findNoneCanntAnnoteDataIDXOnlevelGivenLabel(labels,mod,ontology_level,lb_idx);
			%%-----------------------
			%remove samples that has 0 on first label(due to mistakes make in computing balanced sample which intend to remove labels that are 0
			%% somehow ended in just remove the sample which has 0 on first label )
			idx=findNoneCanntAnnoteDataIDXOnlevel(labels,mod,ontology_level);
			level_labs=all_level_labs(idx,i);
			bowmatrix =featureMat(idx,:);
            dataSetsID_nocAnnot=dataSetsID(idx);
			
			%% use the train& test index generated from blanceing examples on BOW 
            Train=bowmatrix(train_IDX, :);
			Test =bowmatrix(test_IDX,:);
            test_dataSetsID=dataSetsID_nocAnnot(test_IDX);
			clear bowmatrix;
			
			
			%% this is the real implementation that remove samples that have 0 one a given label
			tr_label  =level_labs(train_IDX,:);
			te_label  =level_labs(test_IDX,:);
			
            tr_idx=(tr_label~=0);
			tr_label = tr_label(tr_idx);
            te_idx=(te_label~=0);
			te_label = te_label(te_idx);
            cur_test_dataSetsID =test_dataSetsID(te_idx);
			
           
					[out1, out2, out3, out4]= LinearBinSVM(Train(tr_idx,:), tr_label, Test(te_idx,:),te_label, 1);
                     %[auc_value(i-startR+1),  accu(:,i-startR+1), predict_LBs{i-startR+1}]
						auc_value(i-startR+1)              =out1;
						if size(accu,1) ~=length(out2)
							disp(['size mismatch accu = ' num2str(size(accu,1)) 'and out size =' num2str(length(out2))  ]);
						end
						accu(:,i-startR+1)                 =out2;
						predict_LBs{i-startR+1}            =out3;
						predict_probability{i-startR+1}    =out4;
					% %catch err
					   
					% %end
			% else
				%auc_value(i-startR+1) =0.1;
				%accu(:,i-startR+1)    = zeros(3,1);
				%predict_LBs{i-startR+1} =te_label;
			% end
			%% ============================================================================================================================================
			te_dataSetsIDs{i-startR+1} =cur_test_dataSetsID;
            true_LBs{i-startR+1}= te_label;
			% disp('size a  =');
			% size(a)
			% disp('size b =');
			% size(b)
            
        end
		% if size(accu,2)<length(auc_value);
				% accu(:,length(auc_value))=zeros(size(accu,1),1);
		% end
       OK                   ='TRUE';
       my_mod               =mod_mat(startR:lastR);
       my_ontology          =ontology_mat(startR:lastR);
       my_structName        =structName(startR:lastR);
	   disp(['Machine ' ':' num2str(my_rank) ' finished and sending result to master process']);
	   
	   %save_par_file    =fullfile(temp_dir, ['tempTrainResult_rank_' num2str(my_rank) '.mat']);
	   %save(save_par_file,'auc_value','accu','my_mod','my_ontology');
       MPI_Send ( auc_value, source+1, mytag+4, CW);
       MPI_Send ( accu, source+1, mytag+5, CW);
       MPI_Send ( OK, source+1, mytag+6, CW);
       MPI_Send ( my_mod , source+1, mytag+12, CW);
       MPI_Send ( my_ontology, source+1, mytag+13, CW);
       
     
       MPI_Send (my_structName, source+1, mytag+23, CW);
       MPI_Send (te_dataSetsIDs, source+1, mytag+33, CW);
       MPI_Send (predict_LBs, source+1, mytag+43, CW);
	   MPI_Send (predict_probability, source+1, mytag+44, CW);
       MPI_Send (true_LBs, source+1, mytag+45, CW);
       
      
      
       
    else
       OK       ='FALSE';
       disp('MPI recieve error ....')
       
       MPI_Send ( OK, source+1, mytag+6, CW);
    end
elseif (my_rank ==0)
    matP1=load(source_fileP1);
	matP2=load(source_fileP2);
	matP3=load(source_fileP3);
	matP4=load(source_fileP4);
    % get labels from reading the label file 
    load(labefile);
	
	featureMat=[matP1.Mat' matP2.Mat' matP3.Mat' matP4.Mat']';
	clear matP1.Mat;
	clear matP2.Mat;
	clear matP3.Mat;
	clear matP4.Mat;
	featureMat=sparse(featureMat);
     all_level_labs=[];
     ontology_mat  =[];
     mod_mat       =[];
     structName    ={};
	 %[train_IDX,test_IDX]=find_TrainTest_IDX_from_predefined(dataSetsID);
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
		
        % mod = 'intensity';
        % level_labs_intensity=getSpecifiedLabels(labels,mod,ontology_level);
        % all_level_labs      = [all_level_labs level_labs_intensity];
        % ontology_mat        = [ontology_mat ones(1,size(level_labs_intensity,2))*ontology_level];
        % mod_mat             = [mod_mat ones(1,size(level_labs_intensity,2))*2];
        % structName          = [structName cur_struct_names'];    
        % mod = 'density';
        % level_labs_density  =getSpecifiedLabels(labels,mod,ontology_level);
        % all_level_labs      = [all_level_labs level_labs_density];
        % ontology_mat        = [ontology_mat ones(1,size(level_labs_density,2))*ontology_level];
        % mod_mat             = [mod_mat ones(1,size(level_labs_density,2))*3];
        % structName          = [structName cur_struct_names'];    
		
      % % mod_mat
		%%all_level_labs
    end
	MPI_Send (ontology_mat, 1, mytag+100, CW);
	MPI_Send (mod_mat, 1, mytag+101, CW);
    MPI_Send (structName, 1, mytag+102, CW);
	
	for source =2:p-1
		%disp(['there are '  num2str(sum(ontology_mat~=1)) ' None 1 ontology  to SEND'])
		%disp(['there are '  num2str(sum(mod_mat~=1)) ' None 1 mad  to SEND'])
        MPI_Send (featureMat, source, mytag, CW);
        MPI_Send (all_level_labs, source, mytag+1, CW);
        MPI_Send (ontology_mat, source, mytag+2, CW);
        MPI_Send (mod_mat, source, mytag+3, CW);
		MPI_Send (labels, source, mytag+9,CW);
		MPI_Send (Age, source, mytag+10,CW);
        MPI_Send (structName, source, mytag+20,CW);
        MPI_Send (dataSetsID, source, mytag+30,CW);
		%MPI_Send (train_IDX, source, mytag+40,CW);
		%MPI_Send (test_IDX, source, mytag+50,CW);
       % MPI_Send (labels, source, mytag+4, CW);
    end
    
     
 elseif (my_rank==1)
 
 
		  auc_value=[];
		  accu =[];
		  mod_MPI_order=[];
		  ontology_MPI_order=[];
		  result_save_file =  save_file;
		  if exist(result_save_file,'file')
			 delete(result_save_file);
		  end
		
		 [ontology_mat, info_ontg]           = MPI_Recv (0, mytag+100, CW);
         [mod_mat, info_mat]                = MPI_Recv (0, mytag+101, CW);
         [structName, info_mat]                = MPI_Recv (0, mytag+102, CW);
		 save(fullfile(temp_dir, 'mod_ont.mat'),'mod_mat','ontology_mat');
		 disp('Rec base info from machine 0');
		if (info_mat==MPI_SUCCESS && info_ontg == MPI_SUCCESS)
				for source = 2:p-1
				 % disp ("We are at rank 0 that is master etc..");
				  [auc_r, info1]              = MPI_Recv (source, mytag+4, CW);
				  [accu_r, info2]             = MPI_Recv (source, mytag+5, CW);
				  [OK, info3]                 = MPI_Recv (source, mytag+6, CW);
				  [my_mod, info4]             = MPI_Recv (source, mytag+12, CW);
				  [my_ontology, info5]        = MPI_Recv (source, mytag+13, CW);
                  [my_structName,info6 ]      = MPI_Recv (source, mytag+23, CW);
                  [test_dataSetsID_r,info7]   = MPI_Recv (source, mytag+33, CW);
                  [predict_LBs_r,info8 ]      = MPI_Recv (source, mytag+43, CW);
				  [predict_probs_r,info10 ]   = MPI_Recv (source, mytag+44, CW);
                  % MPI_Send (true_LBs, source+1, mytag+45, CW);
                  [true_LBs_r,info9 ]         = MPI_Recv (source, mytag+45, CW);
				  

				  if (info1 == MPI_SUCCESS && info2 == MPI_SUCCESS && info3 == MPI_SUCCESS && info4 == MPI_SUCCESS && info5 == MPI_SUCCESS  && strcmp(OK,'TRUE')
                      && info6 == MPI_SUCCESS && info7 == MPI_SUCCESS && info8 == MPI_SUCCESS && info9 == MPI_SUCCESS&& info10 == MPI_SUCCESS)
					  disp(['recieved from ' num2str(source)])
					  %auc_value
						  if isempty(auc_value)
							  auc_value             =auc_r;
							  accu                  =accu_r;
							  mod_MPI_order         =my_mod;
							  ontology_MPI_order    =my_ontology;
                              structName_MPI_order   =my_structName;
                              test_dataSetsID       =test_dataSetsID_r;
                              predict_LBs           =predict_LBs_r;
							  predict_probs         =predict_probs_r;
                              true_LBs              =true_LBs_r;
						  else
							  auc_value             =[ auc_value auc_r];
							  accu                  =[accu accu_r];
							  mod_MPI_order         =[mod_MPI_order my_mod];
							  ontology_MPI_order    =[ontology_MPI_order my_ontology];
                              structName_MPI_order   =[structName_MPI_order my_structName];
                              test_dataSetsID       =[test_dataSetsID test_dataSetsID_r];
                              predict_LBs           =[predict_LBs  predict_LBs_r];
							  predict_probs         =[predict_probs predict_probs_r];
                              true_LBs             =[true_LBs true_LBs_r];
                              
						  end
						  %disp(['Size of auc =' num2str(length(auc_value))]);
						  %disp(['Size of auc =' num2str(length(accu))]);
						  %save_par_file         =fullfile(temp_dir, ['tempTrainResult_rank_' num2str(source) '.mat']);
						  %save(save_file,'auc_value','accu','mod_MPI_order','ontology_MPI_order');
						  
				  else
					  disp('master process recieved error ....')
				  end
			   end 
              %save(save_file,'auc_value','ontology_mat','mod_mat','mod_MPI_order','ontology_MPI_order','structName','structName_MPI_order','test_dataSetsID','predict_LBs','true_LBs');
			  save(save_file,'auc_value','accu','ontology_mat','mod_mat','mod_MPI_order','ontology_MPI_order','structName','structName_MPI_order','test_dataSetsID','predict_LBs','true_LBs','predict_probs');
	   end
 end

MPI_Finalize();  
end
function  cur_struct_names =getSpecifiedStructName(structLabels,ontology_lelvel)
          CData =structLabels{ontology_lelvel};
          cur_struct_names = CData(:,1);
end

function  cur_level_labs=getSpecifiedLabels(labels,mod,ontology_level)
    switch mod
        case 'pattern'
            m_idx =1;
        case 'intensity'
            m_idx =2;
        case 'density'
            m_idx =3;
    end
    lb_temp=labels{1};
    level_labels=lb_temp{ontology_level};
    num_labs=size(level_labels,1);
    
    
    cur_level_labs=zeros(length(labels),num_labs);
        
        
    for i=1:length(labels)
        lb=labels{i};
        level_labels=lb{ontology_level};
        %lab_len=size(level_labels,1);
        cur_level_labs(i,:)=level_labels(:, m_idx);
    end
    %lv_lab=labels();
    %lb=lv_lab(:,)
end
function idx=findNoneCanntAnnoteDataIDXOnlevel(labels,mod,ontology_level)
     switch mod
        case 1
            mod_str ='pattern';
        case 2
            mod_str='intensity';
        case 3
            mod_str ='density';
    end
	level_labs=getSpecifiedLabels(labels,mod_str,ontology_level);
	idx=findNoneCanntAnnoteDataIDX(level_labs);
end
function idx=findNoneCanntAnnoteDataIDX(level_labs)
    idx=(level_labs(:,1)~=0);
    %labels=level_labs(idx,:);
    %matrix=matrix(idx,:);
end


function idx=findNoneCanntAnnoteDataIDXOnlevelGivenLabel(labels,mod,ontology_level,lb_idx)
     switch mod
        case 1
            mod_str ='pattern';
        case 2
            mod_str='intensity';
        case 3
            mod_str ='density';
    end
	level_labs=getSpecifiedLabels(labels,mod_str,ontology_level);
	idx=findNoneCanntAnnoteDataIDX_onLb_idx(level_labs,lb_idx);
end
function idx=findNoneCanntAnnoteDataIDX_onLb_idx(level_labs,idx)
    idx=(level_labs(:,idx)~=0);
    %labels=level_labs(idx,:);
    %matrix=matrix(idx,:);
end
