truth_image = translatePath('z:\caffe_flx_kernel\result\truth.tif');
if exist(truth_image)
    delete(truth_image);
end

predict_matrix_file= translatePath('z:\caffe_flx_kernel\data\senms3d_preidct_tr.mat');
load(predict_matrix_file);
D=feat_label.feat;
[h,w,d]=size(D);
if w>1024
    D=D(:,1:1024,:);
end
if d>1024
    D=D(:,:,1:1024);
end
for ind  = 1 :  h
    ind;
    im =squeeze(D(ind,:,:));
    im=uint16(im);
    imwrite(im,truth_image,'WriteMode','append');
    disp(['write', num2str(ind), 'tif']);
end
