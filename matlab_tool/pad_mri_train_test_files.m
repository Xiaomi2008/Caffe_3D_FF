train_mat_file='z:\caffe_3d\data\mri_train.mat';
test_mat_file ='z:\caffe_3d\data\mri_test.mat';


train_mat_pad_file ='z:\caffe_3d\data\mri_train_pad.mat';
test_mat_pad_file ='z:\caffe_3d\data\mri_test_pad.mat';

TR=load(train_mat_file);
data=cellfun(@(x) padarray(x,[6 6 6]), TR.data,'UniformOutput', false);
elm_labels=cellfun(@(x) padarray(x,[6 6 6]), TR.elm_labels,'UniformOutput', false);
labels=TR.labels;
save(train_mat_pad_file,'data','elm_labels','labels','-v7.3');


TE=load(test_mat_file);
data=cellfun(@(x) padarray(x,[6 6 6]), TE.data,'UniformOutput', false);
elm_labels=cellfun(@(x) padarray(x,[6 6 6]), TE.elm_labels,'UniformOutput', false);
labels=TE.labels;
save(test_mat_pad_file,'data','elm_labels','labels','-v7.3');


