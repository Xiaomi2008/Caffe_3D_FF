addpath('/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/matlab/caffe');
addpath('/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/matlab');
model_def_file ='/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/vgg_16_fine_tune_gen/deploy_5paired_images_with_elt_max_featureEx.prototxt';
model_file    ='/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/finetune_vgg_16__ELTmaxpool_multi_shared_weight_net_lastLR_10-20_TLR_1-E4_step2000_blancedSample_ontoL_5_SLICE_ISH_iter_15000';



%model_def_file ='/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/vgg_16_fine_tune_gen/train_dummy_data_shared_weights_net.prototxt';
%model_file    ='/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/dummy_sharedweigt_net_iter_100';




net = CaffeNet.instance(model_def_file,model_file);
%net = CaffeNet.instance
net.set_mode_gpu();
net.set_device(2);
net.set_phase_test;
netweights=net.get_weights;
save('netweights.mat','netweights','-v7.3');

% f1=netweights(3).weights{1};
% f2=netweights(4).weights{1};
% f1=f1(:);
% f2=f2(:);
% ismember(f1',f2','rows')


% c1=netweights(1).weights{1};
% c2=netweights(2).weights{1};
% c1=c1(:);
% c2=c2(:);
% ismember(c1',c2','rows')


