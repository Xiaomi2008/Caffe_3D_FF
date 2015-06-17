function [trainData,valData,testData ] = readCaffeTrainTestImageListFromTextFile(Age,ImageType)
data_dir=translatePath('Z:\ABA\autoGenelable_multi_lables_proj\data');
assert(strcmp(ImageType,'ISH') || strcmp(ImageType,'MSK'))
if strcmp(ImageType,'ISH')
   train_txt='train_ish.txt';
   val_txt  ='val_ish.txt';
   test_txt ='test_ish.txt';
elseif strcmp(ImageType,'MSK')
   train_txt='train_msk.txt';
   val_txt  ='val_msk.txt';
   test_txt ='test_msk.txt';
end

train_output_file =     [data_dir filesep train_txt];
val_output_file   =     [data_dir filesep val_txt];
test_output_file  =     [data_dir filesep test_txt];

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

