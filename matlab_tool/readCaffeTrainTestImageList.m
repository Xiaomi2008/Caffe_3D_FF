function [trainData,valData,testData ] = readCaffeTrainTestImageList()
data_dir=translatePath('z:\ABA\automatedGeneLabelingProject\manannot\data');
train_output_file =     [data_dir filesep 'train.txt'];
val_output_file   =     [data_dir filesep 'val.txt'];
test_output_file  =     [data_dir filesep 'test.txt'];

trainStruct=importdata(train_output_file);
trainData.labels    =trainStruct.data;
trainData.imageFile =trainStruct.textdata;

valStruct=importdata(val_output_file);
valData.labels    =valStruct.data;
valData.imageFile =valStruct.textdata;


testStruct=importdata(test_output_file);
testData.labels    =testStruct.data;
testData.imageFile =testStruct.textdata;

end

