
function generateCNN_multilabel_multiclass_protoFile(outputFileName)
bottom_blob_name='fc7';
top_blob_name='fc8'
full_connection_layer_name=top_blob_name
fileID = fopen(outputFileName,'w');
blobs_lr_1=3;
blobs_lr_2=6;
num_class =4;
label_name ='L';
num_labels=81;



% for i=1:num_labels
	% fprintf(fileID,'%s{\n','layers');
	% fprintf(fileID,'bottom: "%s"\n',bottom_blob_name);
	% fprintf(fileID,'top: "%s_%d"\n',top_blob_name,i);
	% fprintf(fileID,'name: "%s_%d"\n',top_blob_name,i);
	% fprintf(fileID,'type: %s\n','INNER_PRODUCT');
	% fprintf(fileID,'blobs_lr: %d\n',blobs_lr_1);
	% fprintf(fileID,'blobs_lr: %d\n',blobs_lr_2);
	% fprintf(fileID,'weight_decay: %d\n',1);
	% fprintf(fileID,'weight_decay: %d\n',0);
	% fprintf(fileID,'inner_product_param {\n');
	% fprintf(fileID, '	num_output: %d\n',num_class);
	% fprintf(fileID,'	weight_filler {\n');
	% fprintf(fileID,'		type: "%s" \n','gaussian');
	% fprintf(fileID,'		std: %5.2f\n',0.01);
	% fprintf(fileID,'		}\n');
	% fprintf(fileID,'	bias_filler {\n');
	% fprintf(fileID,'		type: "%s" \n','constant');
	% fprintf(fileID,'		value: %d \n',0);
	% fprintf(fileID,'		}\n');
	% fprintf(fileID,'	}\n');
	% fprintf(fileID,'}\n\n');
% end


% for i=1:num_labels
	% fprintf(fileID,'%s{\n','layers');
	% fprintf(fileID,' name: "%s_%d"\n','loss',i);
	% fprintf(fileID,' type: %s\n',' SOFTMAX_LOSS');
	% fprintf(fileID,' bottom: "%s_%d"\n',top_blob_name,i);
	% fprintf(fileID,' bottom: "%s%d"\n',label_name,i-1);
	% fprintf(fileID,'}\n\n');

% end

for i=1:num_labels
	fprintf(fileID,'%s{\n','layers');
	fprintf(fileID,' name: "%s_%d"\n','accuracy',i);
	fprintf(fileID,' type: %s\n',' ACCURACY');
	fprintf(fileID,' bottom: "%s_%d"\n',top_blob_name,i);
	fprintf(fileID,' bottom: "%s%d"\n',label_name,i-1);
	fprintf(fileID,' top:    "%s_%d"\n','accuracy',i)
	fprintf(fileID,'}\n\n');
end



% layers {
  % name: "accuracy_1"
  % type: ACCURACY
  % bottom: "fc8_1"
  % bottom: "L0"
  % top: "accuracy_1"
% }



% layers {
  % name: "loss_1"
  % type: SOFTMAX_LOSS
  % bottom: "fc8_1"
  % bottom: "L0"
% }



% layers {
  % bottom: "fc7"
  % top: "fc8_1"
  % name: "fc8_1"
  % type: INNER_PRODUCT
  % blobs_lr: 3
  % blobs_lr: 6
  % weight_decay: 1
  % weight_decay: 0
  % inner_product_param {
    % num_output: 4
	% weight_filler {
      % type: "gaussian"
      % std: 0.01
    % }
    % bias_filler {
      % type: "constant"
      % value: 0
    % }
  % }
% }
end