export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda-6.5/lib64:/usr/local/lib
GLOG_logtostderr=1 ../build/tools/fast_image_label_predict.bin ../data/snems3d_predict_norm  ../models/deploy_snems3d_as2d_2.prototxt ../models/snems3d_pad_2_47_47_2D_deeper_bg_downsampled_fine_iter_124000 5 1118 1118 SINGLE 50   prob ../data/fast_snems3d_predict  GPU 1 MEAN_VALUE 0



#GLOG_logtostderr=1 ../build/tools/fast_image_label_predict.bin ../data/snems3d_train_pad_3_47_47_cpy ../models/deploy_snems3d.prototxt ../models/snems3d_pad_3_37_47_conv_iter_200  9 1118 1118 prob ../data/fast_snems3d_predict

python ../tools/convert_lDB2mat_label_predict.py --feature_db ../data/fast_snems3d_predict --mat ../data/senms3d_preidct_tr.mat
#snems3d_predict_pad_3_47_47
#snems3d_predict_norm_pad_4_47_47
#snems3d_train_pad_3_47_47_cpy
#snems3d_pad_2_47_47_2D_deeper_bg_downsampled_iter_8000
#snems3d_test_norm_cpy 
#snems3d_predict_norm
