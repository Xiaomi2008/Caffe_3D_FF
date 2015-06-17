freq_file='z:\caffe_flx_kernel\data\snems3d_class_freq.mat';
protxt_file='z:\caffe_flx_kernel\models\snems3d_label_class_selection.protxt';
load(freq_file);
fid=fopen(protxt_file,'w');

len= length(count);
for i=1:len
    fprintf(fid,'%s\n','  label_prob_mapping_info{');
    fprintf(fid,'%s%d\n', '         label: ',label(i));
    fprintf(fid,'%s%f\n', '         prob: ',prob(i));
    %fprintf(fid,'%s%f\n', '  prob: ',prob(i));
    fprintf(fid,'%s\n',' }');
end
fclose(fid);