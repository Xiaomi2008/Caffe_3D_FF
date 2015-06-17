ImageType=ISH
Age=E11.5
#Layer=conv5_1

#####################################
#For 5 paired images, a element max layer is created at following layers, so that it can extract element max from 5 inputs
#Layer=conv5_1_concat;
Layer=flat_conv5_3_eltmax;
#2. flat_conv5_2_eltmax
#3. flat_conv5_3_eltmax
################################
DeviceID=2
#featureType=pairedImage
featureType=multiImage
#singleImage 

#pairedImage #multiImage 

# using ethier maksed or original ISH image corresponding, Note that the fine tuned model has to be trained correspondingly.



# set which steps to run ----------------
run_set_param=true
run_extract_im_feature=true
#run_build_section_feature=false
run_build_sections_martix=true
run_svm=true
#------------------------------------------


export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda-6.5/lib64
codehome=/home/tzeng/caffe_3d/code/matlab_tool
cd $codehome


model_dir=/home/tzeng/caffe_3d/models

temp_dir=/home/tzeng/caffe_3d/temp


#using finTune Model with 10 multiple image slices with shared weight to finTune the networks #####################
model_def_file=$model_dir/deploy_ELTmaxpool_multi_sharedwieght_net_slice10_ish_1.prototxt
#mf_name=finetune_vgg_16__ELTmaxpool_multi_shared_weight_net_10_SLICE_ISH_4500_resume_iter_1500


## single image trained but extracting feature by above 10 slice prototxt to get element max over each image's feature
mf_name=finetune_vgg_16_lastLR_3-6_TLR_3-E5_step2500_blancedSample_ontoL_5_ISH_iter_15000
model_file=$model_dir/$mf_name
##################################################




inf_exchange_file_name=$temp_dir/infor_exchange_$mf_name$Age$Layer$ImageType










# 1. set ans save param for runing the pipline.
if [ "$run_set_param" = true ]; then
	temp_commd1=$codehome/comd1.m
	echo "set_run_all_param('$ImageType','$model_def_file','$model_file','$Layer','$Age','$inf_exchange_file_name',$DeviceID,'$featureType');">$temp_commd1
	echo "exit">>$temp_commd1
	#param="set_run_all_param($model_def_file,$model_file);quit;";
	matlab -r "comd1"
fi



#2. extract feature image level feature
if [ "$run_extract_im_feature" = true ];then

    temp_commd2=$codehome/comd2.m
	echo "run_extract_image_feature('$inf_exchange_file_name');">$temp_commd2
	echo "exit">>$temp_commd2
	matlab -r "comd2"
	#matlab -r "run_extract_image_feature;quit;"
fi



#3. to build section level feature matrix on hpcd using MPI accelation
if [ "$run_build_section_feature" = true ];then
temp_commd3=$codehome/comd3.m
echo "run_build_section_feature('$inf_exchange_file_name');">$temp_commd3
echo "exit">>$temp_commd3
ssh tzeng@hpcd.cs.odu.edu<<'ENDSSH'
cd /home/tzeng/ABA/autoGenelable_multi_lables_proj/code/matlab
hostname
matlab -r "comd3"
ENDSSH
#mpirun -np 32 -bycore -machinefile hostname octave --eval run_build_caffe_sectionFeature_octave_MPI('$inf_exchange_file_name')
#matlab -r "run_build_section_feature;quit;"
fi





#4.  build all section features into single matrix;
if [ "$run_build_sections_martix" = true ];then
temp_commd4=$codehome/comd4.m
echo "run_build_AllsectionFeature2Mat('$inf_exchange_file_name');">$temp_commd4
echo "exit">>$temp_commd4
#matlab -r "comd4"
#matlab -r "run_build_AllsectionFeature2Mat;quit;"

ssh tzeng@sirius.cs.odu.edu<<'ENDSSH'
cd /home/tzeng/ABA/autoGenelable_multi_lables_proj/code/matlab
hostname
matlab -r "comd4"
ENDSSH



fi



#5. train and test svm on section level features
# ssh tzeng@hpcd.cs.odu.edu<<'ENDSSH'
# cd /home/tzeng/ABA/autoGenelable_multi_lables_proj/code/matlab
# matlab -r "run_svm_train;quit;"
# ENDSSH


#5. train and test svm on section level features
if [ "$run_svm" = true ];then
temp_commd5=$codehome/comd5.m
echo "run_svm_train('$inf_exchange_file_name');">$temp_commd5
echo "exit">>$temp_commd5
#ssh tzeng@sirius.cs.odu.edu<<'ENDSSH'
ssh tzeng@hpcd.cs.odu.edu<<'ENDSSH'
cd /home/tzeng/ABA/autoGenelable_multi_lables_proj/code/matlab
hostname
matlab -r "comd5"
ENDSSH
fi
#matlab -r "run_svm_train;quit;"

# # #
# #matlab -r "pipeline_run_all;quit;"