function run_build_setion_feature(inf_exchange_file_name)
	tempDir=translatePath('Z:\ABA\autoGenelable_multi_lables_proj\temp'); 
	addpath('/home/tzeng/ABA/automatedGeneLabelingProject/code');
	info_exchange_file =[inf_exchange_file_name '.mat'];
	%MPI_info_exchange_file =[tempDir filesep MPI_info_exchange_file];
	%load(MPI_info_exchange_file);
	MPI_info_exchange_file =[tempDir filesep 'MPI_age_layer_info.mat']
	copyfile(info_exchange_file, MPI_info_exchange_file,'f');
	nump =32;
	%ss=['' MPI_info_exchange_file ''];
	cmd =['mpirun -np ' num2str(nump) ' -bycore -machinefile hostname octave run_build_caffe_sectionFeature_octave_MPI.m']
	system(cmd);
	
	
end