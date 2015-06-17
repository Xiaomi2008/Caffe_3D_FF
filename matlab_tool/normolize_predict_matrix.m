if ispc
	train_mat_file='z:\caffe_flx_kernel\data\snems3d_test.mat';
	train_mat_pad_file ='z:\caffe_flx_kernel\data\snems3d_predict_norm.mat';
else
	train_mat_file='/home/tzeng/caffe_flx_kernel/data/snems3d_test.mat';
	train_mat_pad_file ='/home/tzeng/caffe_flx_kernel/data/snems3d_predict_norm.mat';
end
TR=load(train_mat_file);
%data=cellfun(@(x) padarray(x,[pad_h pad_w pad_d]), TR.data,'UniformOutput', false);
%elm_labels=cellfun(@(x) padarray(x,[pad_h pad_w pad_d]), TR.elm_labels,'UniformOutput', false);
%labels=TR.labels;


d2=TR.data{1};
[h,w,d]=size(d2);
l=TR.elm_labels{1};
%DFLR=zeros(h,w,d);
%LFLR=zeros(h,w,d);
D=zeros(h,w,d);
for i=1:h
    disp(['procesing ', num2str(i), ' rows']);
	%dt=d(i,:,:);
    d1=squeeze(d2(i,:,:));
    d1=mat2gray(d1);
    d1=single(uint8(d1*255));
    
    %l1=squeeze(l(i,:,:));
    D(i,:,:)=d1;
    %DFLR(h-i+1,:,:) =flipud(d1);
    %LFLR(h-i+1,:,:) =flipud(l1);
end


data{1}=D;
elm_labels{1}=single(l);

disp(['saving mat file to ',train_mat_pad_file]);
save(train_mat_pad_file,'data','elm_labels','-v7.3');