if ispc
	train_mat_file='z:\caffe_flx_kernel\data\snems3d_test.mat';
	train_mat_pad_file ='z:\caffe_flx_kernel\data\snems3d_predict_pad.mat';
else
	train_mat_file='/home/tzeng/caffe_flx_kernel/data/snems3d_test.mat';
	train_mat_pad_file ='/home/tzeng/caffe_flx_kernel/data/snems3d_preidct_pad.mat';
end
pad_h=3;
pad_w=47;
pad_d=47;
TR=load(train_mat_file);
data=cellfun(@(x) padarray(x,[pad_h pad_h pad_w]), TR.data,'UniformOutput', false);
elm_labels=cellfun(@(x) padarray(x,[pad_h pad_w pad_d]), TR.elm_labels,'UniformOutput', false);
%labels=TR.labels;
save(train_mat_pad_file,'data','elm_labels','-v7.3');