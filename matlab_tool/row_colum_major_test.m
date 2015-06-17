%d=feat_label.label(20,:);
%d=elm_labels{20};
%d=data{20};
%d_s=reshape(d,[54 53 62]);
%d_s_p=permute(d_s,[3 2 1]);
%vol3d('cdata',d_s_p);
%d_s_p_f=d_s_p(:);

figure

d2=feat_label.feat(10,:);
%d2(d2<0)=0;
d2_s_p=reshape(d2,[50 41 42]);
%d2_s_p=reshape(d2,[42 41 50]);
%d2_p=permute(d2_s_p,[3 2 1]);

vol3d('cdata',d2_s_p);


figure
l2=feat_label.label(10,:);
%l2=permute(l2,[3 2 1])
l2_s_p=reshape(l2,[50 41 42]);
%l2_s_p=reshape(l2,[42 41 50]);
vol3d('cdata',l2_s_p);


d_im=data{10};

im1_idx=19;
im2_idx=21;
im3_idx=23;
im4_idx=26;
im5_idx=29;
figure
 subplot(5,3,1),imshow(squeeze(d2_s_p(:,:,im1_idx))');
 subplot(5,3,2),imshow(squeeze(l2_s_p(:,:,im1_idx))');
 subplot(5,3,3),imshow(squeeze(d_im(im1_idx,:,:))/15);
 
 subplot(5,3,4),imshow(squeeze(d2_s_p(:,:,im2_idx))');
 subplot(5,3,5),imshow(squeeze(l2_s_p(:,:,im2_idx))');
 subplot(5,3,6),imshow(squeeze(d_im(im2_idx,:,:))/15);
 
 subplot(5,3,7),imshow(squeeze(d2_s_p(:,:,im3_idx))');
 subplot(5,3,8),imshow(squeeze(l2_s_p(:,:,im3_idx))');
 subplot(5,3,9),imshow(squeeze(d_im(im3_idx,:,:))/15);
 
 
 subplot(5,3,10),imshow(squeeze(d2_s_p(:,:,im4_idx))');
 subplot(5,3,11),imshow(squeeze(l2_s_p(:,:,im4_idx))');
 subplot(5,3,12),imshow(squeeze(d_im(im4_idx,:,:))/15);
 
 subplot(5,3,13),imshow(squeeze(d2_s_p(:,:,im5_idx))');
 subplot(5,3,14),imshow(squeeze(l2_s_p(:,:,im5_idx))');
 subplot(5,3,15),imshow(squeeze(d_im(im5_idx,:,:))/15);


 

 
 

%figure

%d2_s_p=permute(d2_s_p,[3 2 1]);
%d2_s_p_f=d2_s_p(:);

%dd=reshape(d2_s_p_f,[42 41 50]);
%vol3d('cdata',dd);
%d_s_p_f=d_s_p(:);


% d_s_p_f_s=reshape(d_s_p_f,[41 42 50]);
% d_s_p_f_s_p=permute(d_s_p_f_s,[3 2 1]);
% vol3d('cdata',d_s_p_f_s_p);
% d_size=size(d);
% d_p=permute(d,[3 2 1]);
% d_p_f=d_p(:);
% d_p_f_s=reshape(d_p_f,d_size);
% figure
% vol3d('cdata',d_p_f_s);
% 
% d_p_f_s_s=reshape(d_p_f_s,d_size);
% figure
% vol3d('cdata',d_p_f_s_s);
% 
% 
% d_p_f_s_s_p=permute(d,[3 2 1]);
% figure
% vol3d('cdata',d_p_f_s_s_p);