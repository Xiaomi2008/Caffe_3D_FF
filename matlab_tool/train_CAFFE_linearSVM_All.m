function train_CAFFE_linearSVM_All(Age,mat_file)
addpath('/home/tzeng/ABA/automatedGeneLabelingProject/code');
data_dir=translatePath('Z:\ABA\autoGenelable_multi_lables_proj\data');
temp_dir=translatePath('Z:\ABA\autoGenelable_multi_lables_proj\temp');
% if nargin <6
% host ='hpcd';
% else
% host ='hpcq';
% end
caffe_mex_file=fullfile(data_dir,[mat_file '.mat']);
saveDir=translatePath('Z:\ABA\autoGenelable_multi_lables_proj\result');
%filePartMod='whole';
%featuresampleMod ='Max';

saveFile =[saveDir  filesep 'classResultMIP_' mat_file '.mat'];



if ~exist(caffe_mex_file,'file')
    disp(['File : ' caffe_mex_file ' not exist quit ...'])
    return;
else
    load(caffe_mex_file);
end

if(max(CAFFE_Matrix(:))==0)
disp('Coffe_matrix all 0');
disp(caffe_mex_file);
end

norm_matrix=norm_bow_matrix(CAFFE_Matrix);

if(sum((norm_matrix(:)~=0))==0)
disp('norm_matrix all 0');
end

clear CAFFE_Matrix;
temp_dir =translatePath('Z:\ABA\autoGenelable_multi_lables_proj\temp');
labefile=fullfile(temp_dir, 'tempTrainFeatures_labels.mat');
%save(labefile,'labels', '-v6'); 
save(labefile,'labels', 'Age','dataSetsID', 'structureName','-v7');





%system(['mpirun -np ' num2str(nump) ' -machinefile hostname octave run_cur_OFV_train.m']);
result_file       =fullfile(temp_dir, 'tempTrainResult.mat');
%while ~exist(result_file,'file')
    [~,b]=system('hostname');
	if strcmp(b(1:4),'hpcd')
	    partitionAndSaveMat4(norm_matrix);
        clear norm_matrix;
	    disp(['run train svm on cluster MPI ' b])
		nump =29;
		system(['mpirun -np ' num2str(nump)  ' -bycore -machinefile hostname octave run_cur_CAFFE_train.m']);
		run_on_mpi=true;
	elseif strcmp(b(1:4),'hpcq')
	     partitionAndSaveMat4(norm_matrix);
         clear norm_matrix;
	     disp(['run train svm on cluster MPI ' b]);
		 nump=16
		 system(['mpirun -np ' num2str(nump)  ' -bycore -machinefile hostnamehpcq octave run_cur_CAFFE_train.m']);
	else
	    
	     disp(['run train svm on ' b]);
		caffeData_linear_trainPredict_ontology_5_parfor(norm_matrix);
		run_on_mpi=false;
    end
%end


	if exist(result_file,'file')==2
		load(result_file);
		auc =auc_value;
		acc =accu;
		if run_on_mpi
		   
			save(saveFile,'auc','acc','ontology_mat','mod_mat','mod_MPI_order','ontology_MPI_order','structName','structName_MPI_order','test_dataSetsID','predict_LBs','true_LBs','predict_probs');
			%elseif strcmp(host,'hpcq')
			
		else
		    save(saveFile,'auc','acc','ontology_mat','mod_mat','structName','test_dataSetsID','predict_LBs','true_LBs','predict_probs');	
		end
        %save(saveFile,'auc','ontology_mat','mod_mat','mod_MPI_order','ontology_MPI_order','structName','structName_MPI_order','test_dataSetsID','predict_LBs','true_LBs');
		%if ~fineTune
			saveFilePrefix =[saveDir filesep 'classfied_CAFFE_' mat_file];
		%else
		%end
		disp(['result file saved to ' saveFile ]);
		convertMPI_TrainResult_byOntology(saveFile, saveFilePrefix)
		generateAUCReportOnCAFFE(['classfied_CAFFE_' mat_file]);
	else
		disp(['MPI AUC computation failed to generate file for : ' Age ' Layer : ' num2str(layer)] );
	end
end
%% partition data into 4 parts so that Octave can load small file into memory and cancatenate them into one matrix for MPI computation
function partitionAndSaveMat4(norm_matrix)
        temp_dir =translatePath('Z:\ABA\autoGenelable_multi_lables_proj\temp');
        len =size(norm_matrix,1);
        Tr_oneP =floor(len/4);
        range=0:4;
        Tr_range=range*Tr_oneP;
        Tr_range(1)=1;
        Tr_range(end)=len+1;
        for i=1:4
             cur_idx=Tr_range(i):Tr_range(i+1)-1;
             Mat       =   norm_matrix(cur_idx,:);
             Mat_for_ocvte_MPI_file       =fullfile(temp_dir, ['tempTrainFeaturesMatP' num2str(i) '.mat']);
             save(  Mat_for_ocvte_MPI_file,'Mat', '-v6');      
        end
end

function  matrix=norm_bow_matrix(matrix)
   % [r,c] =size(matrix);
   % minx=min(matrix);
   % maxx=max(matrix);
   % for i=1:c
       % matrix(:,i)=(matrix(:,i)-minx(i))./(maxx(i)-minx(i));
   % end
   % for i = 1: length(index_zero)
    % tr_matrix(:, index_zero(i)) = zeros( size(tr_matrix,1), 1);
   % end
   
   
max_value = 1;
min_value = 0;

max_feat = max(matrix);
min_feat = min(matrix);

diff = max_feat - min_feat;
index_zero = find(diff == 0);

for i = 1:size(matrix,2)
    matrix(:,i) = ((matrix(:,i)-min_feat(i))/(max_feat(i)-min_feat(i)))*(max_value-min_value)+min_value;
%     te_matrix(:,i) = ((te_matrix(:,i)-min_feat(i))/(max_feat(i)-min_feat(i)))*(max_value-min_value)+min_value;
    
end

for i = 1: length(index_zero)
    matrix(:, index_zero(i)) = zeros( size(matrix,1), 1);
end
   
   
end


function  cur_level_labs=getSpecifiedLabels(labels,mode,ontology_level)
    switch mode
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
