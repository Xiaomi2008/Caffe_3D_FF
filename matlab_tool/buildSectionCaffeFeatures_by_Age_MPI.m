function buildSectionCaffeFeatures_by_Age_MPI(ImageType,model_name,Age,layer,sampleMode)
pkg load mpi;
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


if my_rank==0
	serialDirs = get_serial_section_dirs(Age);
	
    for source =1:p-1
	  MPI_Send(serialDirs,source,mytag,CW);
    end
else
    [serialDirs ,info]=MPI_Recv(0,mytag,CW);
	if (info == MPI_SUCCESS)
	    numSections=length(serialDirs);
		range = floor( numSections /(p-1))
		stepping = 0:p-1;
		stepping = stepping * (range);
		stepping(1) = 1;
		stepping(end) =  numSections + 1;
		startR = stepping(my_rank);
		lastR = stepping(my_rank+1)-1;


		for i=startR: lastR
			cur_dir =serialDirs{i};
			if ~ exist(cur_dir,'dir')
				continue;
			end
			%buildSectionOverFeatFeature(cur_dir,imsize,layer, 'whole',sampleMode)
			%buildSectionCaffeFeature(finetune,cur_dir,layer,sampleMode,'whole');
			buildSectionCaffeFeature(ImageType,model_name,cur_dir,layer,sampleMode,'whole')
		end
	end
end

MPI_Finalize();  
end
