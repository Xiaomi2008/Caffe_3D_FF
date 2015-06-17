if ispc
	train_mat_file='z:\caffe_flx_kernel\data\snems3d_train.mat';
	train_mat_pad_file ='z:\caffe_flx_kernel\data\snems3d_train_RF8.mat';
else
	train_mat_file='/home/tzeng/caffe_flx_kernel/data/snems3d_train.mat';
	train_mat_pad_file ='/home/tzeng/caffe_flx_kernel/data/snems3d_train_RF8.mat';
end
TR=load(train_mat_file);
%data=cellfun(@(x) padarray(x,[pad_h pad_w pad_d]), TR.data,'UniformOutput', false);
%elm_labels=cellfun(@(x) padarray(x,[pad_h pad_w pad_d]), TR.elm_labels,'UniformOutput', false);
%labels=TR.labels;


% rot90 and norm

d2=uint8(TR.data{1});
[h,w,d]=size(d2);
l=TR.elm_labels{1};
D=zeros(h,w,d);
D90=zeros(h,w,d);
D180=zeros(h,w,d);
D270=zeros(h,w,d);
L90=zeros(h,w,d);
L180=zeros(h,w,d);
L270=zeros(h,w,d);
RD =zeros(h,w,d);
RL =zeros(h,w,d);
RD90 =zeros(h,w,d);
RL90 =zeros(h,w,d);

DFLR =zeros(h,w,d);
LFLR =zeros(h,w,d);
DFLC =zeros(h,w,d);
LFLC =zeros(h,w,d);
for i=1:h
    disp(['procesing ', num2str(i), ' rows']);
	%dt=d(i,:,:);
    d1=squeeze(d2(i,:,:));
    d1=mat2gray(d1);
    d1=d1*255;
    d90=rot90(d1);
    d180=rot90(d1,2);
    d270=rot90(d1,3);
    l1=squeeze(l(i,:,:));
    l90=rot90(l1);
    l180=rot90(l1,2);
    l270=rot90(l1,3);
    
    D(i,:,:)=d1;
    D90(i,:,:)=d90;
    D180(i,:,:)=d180;
    D270(i,:,:)=d270;
    L90(i,:,:)=l90;
    L180(i,:,:)=l180;
    L270(i,:,:)=l270;
    
    RD(h-i+1,:,:)=d1;
    RL(h-i+1,:,:)=l1;
    RD90(h-i+1,:,:)=d90;
    RL90(h-i+1,:,:)=l90;
    
    DFLR(i,:,:) =fliplr(d1);
    LFLR(i,:,:) =fliplr(l1);
    DFLC(i,:,:) =flipud(d1);
    LFLC(i,:,:) =flipud(l1);
end

data{1}=D;
elm_labels{1}=l;

data{2}=D90;
elm_labels{2}=L90;

data{3}=D180;
elm_labels{3}=L180;

data{4}=D270;
elm_labels{4}=L270;

data{5}=RD;
elm_labels{5}=RL;

data{6}=RD90;
elm_labels{6}=RL90;

data{7}=DFLR;
elm_labels{7}=LFLR;

data{8}=DFLC;
elm_labels{8}=LFLC;

disp(['saving mat file to ',train_mat_pad_file]);
save(train_mat_pad_file,'data','elm_labels','-v7.3');


