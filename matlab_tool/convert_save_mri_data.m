if ispc
data_dir ='z:\caffe_3d\data';
else
 data_dir='/home/tzeng/caffe_3d/data';
end
data_file=[data_dir filesep 'MRI_4D_398_cutmargin.mat'];
load(data_file)
num_sample=size(data,1);
label=label-2;
orig_data=data;
orig_label=label;


pet_data_file=[data_dir filesep 'PET_4D_398_cutmargin.mat'];
pet_data=load(pet_data_file);

for i=1:num_sample

 dtemp=single(squeeze(data(i,:,:,:)));
 all_data{i}=permute(dtemp, [3 2 1]);
 eld_temp=single(squeeze(pet_data.data(i,:,:,:)));
 all_elm_label{i}=permute(eld_temp, [3 2 1]);
end
idx=randperm(num_sample);
all_data=all_data(idx);
all_elm_label=all_elm_label(idx);
all_label=label(idx);

train_data=all_data(1:319);
train_elm_label=all_elm_label(1:319);
train_label=all_label(1:319);

test_data=all_data(320:end);
test_elm_label=all_elm_label(320:end);
test_label=all_label(320:end);


save_train_file=[data_dir filesep 'mri_train.mat'];

save_test_file=[data_dir filesep 'mri_test.mat'];

data = train_data;
elm_labels= train_elm_label;
labels= train_label';

save(save_train_file,'data','elm_labels','labels', '-v7.3');


data = test_data;
labels= test_label';
elm_labels= test_elm_label;
save(save_test_file,'data','elm_labels','labels', '-v7.3');