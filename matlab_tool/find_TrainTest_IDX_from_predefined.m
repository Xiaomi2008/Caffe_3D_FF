function [train_idx,test_idx]=find_TrainTest_IDX_from_predefined(datasetIDS,Age)
 trainTest_partition_files=translatePath(['Z:\ABA\autoGenelable_multi_lables_proj\data\train_validataion_test_set_for_caffeCNN_level7_' Age '.mat']);
 load(trainTest_partition_files);
 C={train_serial.serialNo};
 trainIDsfromFile=cellfun(@(x) str2num(x),C);
 [~,train_idx,id]=intersect(datasetIDS,trainIDsfromFile);
 
 C={test_serial.serialNo};
 testIDsfromFile=cellfun(@(x) str2num(x),C);
 [~,test_idx,id]=intersect(datasetIDS,testIDsfromFile);
 
 
 C={val_serial.serialNo};
 valIDsfromFile=cellfun(@(x) str2num(x),C);
 [~,val_idx,id]=intersect(datasetIDS,valIDsfromFile);
 
 
 size(test_idx)
 size(val_idx)
 if size(test_idx,2)==1
    test_idx=test_idx';
end
if size(val_idx,2)==1
   val_idx=val_idx';
end


 test_idx=[test_idx val_idx];
 
end