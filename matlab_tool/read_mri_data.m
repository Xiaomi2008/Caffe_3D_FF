data_dir ='\home\tzeng\caffe_3d\data';
mri_data_dir =translatePath([data_dir filesep  'combineMRI_PET']);
mri_dir_struct=dir(mri_data_dir );
mri_dir_struct=mri_dir_struct(3:end);
dir_len= length(mri_dir_struct);
count =1;
for i=1:dir_len
   if mri_dir_struct(i).isdir
		dir_name = mri_dir_struct(i).name;
		if strcmp(lower(dir_name), 'normal')
		    label =0;
	    elseif strcmp(lower(dir_name), 'ad')
		    label =1 ;
		elseif strcmp(lower(dir_name), 'pmci')
		    label =2 ;
		elseif strcmp(lower(dir_name), 'smci')
		    label =3 ;
		end
			
		sub_dir_struct=dir(translatePath([mri_data_dir filesep dir_name]));
		sub_dir_struct=sub_dir_struct(3:end);
		for j=1:length(sub_dir_struct)
		  if sub_dir_struct(j).isdir
                 cur_dir =[mri_data_dir filesep dir_name filesep sub_dir_struct(j).name];
		        files_struct = dir(cur_dir); 
                files_struct = files_struct(3:end);
				isfiles=(files_struct.isdir);
                files_struct(isfiles)=[];
				files_names=files_struct;
				for k=1:length(files_names)
                    file=files_names(k).name;
                    ext=file(end-3:end);
				    if sum(strfind(file, 'WM'))>0 && strcmp(ext,'.hdr')
						info = analyze75info([cur_dir filesep file]);
						data{count} = analyze75read(info);
						labels(count) =label;
                        count=count+1;
					end
					
				end

		  end
			
		end
	end
	
end