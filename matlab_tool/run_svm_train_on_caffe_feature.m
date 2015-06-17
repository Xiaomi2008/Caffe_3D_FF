function run_svm_train_on_caffe_feature(trainMatFile,testMatFile,savefile)
tr_dat=load(trainMatFile);
te_dat=load(testMatFile);

tr_dat=tr_dat.feat_label;
te_dat=te_dat.feat_label;
%disp(['saving result file is  '  savefile])
trainPredict_on_caffe_featuer_parfor(tr_dat,te_dat,savefile)

end