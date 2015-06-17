function generate_shared_weight_10_slice_protext_file(filename)

fileID = fopen(filename,'w');
%blobs_lr_1=3;
%blobs_lr_2=6;
num_of_shared_net =10;
net_name ='VGG_16_layers_Brain_annotation';

    fprintf(fileID,'%s{\n','layers');
	fprintf(fileID,' name: "%s"\n','data');
	fprintf(fileID,' type: %s\n',' DATA');
	fprintf(fileID,' top: "%s"\n','data');
	fprintf(fileID,' top: "%s"\n','label');
	fprintf(fileID,' data_param %s\n','{');
	fprintf(fileID,'	source: "%s"\n','/home/tzeng/ABA/autoGenelable_multi_lables_proj/data/train_slice_lvdb_ish');
	fprintf(fileID,'	mean_file: "%s"\n','/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/vgg_16/vgg_mean_224x224_30CH.binaryproto');
	fprintf(fileID,' 	batch_size: %d\n',3);
	fprintf(fileID,' 	crop_size: %d\n',0);
	fprintf(fileID,' 	mirror: %s\n','false');
	fprintf(fileID,' %s\n','}');
	fprintf(fileID,'%s\n\n','}');
	
	fprintf(fileID,'%s{\n','layers');
	fprintf(fileID,'  name: "%s"\n','slice');
    fprintf(fileID,'  type: %s\n','SLICE');
	fprintf(fileID,'  bottom: "%s"\n','data');
	fprintf(fileID,'  top: "%s"\n','data_1');
	fprintf(fileID,'  top: "%s"\n','data_1p');
	fprintf(fileID,'  top: "%s"\n','data_2p');
	fprintf(fileID,'  top: "%s"\n','data_3p');
	fprintf(fileID,'  top: "%s"\n','data_4p');
	fprintf(fileID,'  top: "%s"\n','data_5p');
	fprintf(fileID,'  top: "%s"\n','data_6p');
	fprintf(fileID,'  top: "%s"\n','data_7p');
	fprintf(fileID,'  top: "%s"\n','data_8p');
	fprintf(fileID,'  top: "%s"\n','data_9p');
	fprintf(fileID,'  slice_param %s\n','{');
	fprintf(fileID,'       slice_dim: %d\n',1);
    fprintf(fileID,'   %s\n','}');	
	fprintf(fileID,'  %s\n','}');


for i=1:num_of_shared_net
	if i==1
	  bottom_data='data_1';
	  conv1_1name   ='conv1_1';
	  relu1_1name   ='relu1_1'
	  conv1_2name   ='conv1_2';
	  relu1_2name   ='relu1_2'
	  pool1_name    ='pool1';
	  conv2_1name   ='conv2_1';
	  relu2_1name   ='relu2_1';
	  conv2_2name   ='conv2_2';
	  relu2_2name   ='relu2_2';
	  pool2_name    ='pool2';
	  conv3_1name   ='conv3_1';
	  relu3_1name   ='relu3_1';
	  conv3_2name   ='conv3_2';
	  relu3_2name   ='relu3_2';
	  conv3_3name   ='conv3_3';
	  relu3_3name   ='relu3_3';
	  pool3_name    ='pool3';
	  conv4_1name   ='conv4_1';
	  relu4_1name   ='relu4_1';
	  conv4_2name   ='conv4_2';
	  relu4_2name   ='relu4_2';
	  conv4_3name   ='conv4_3';
	  relu4_3name   ='relu4_3';
	  pool4_name    ='pool4';
	  conv5_1name   ='conv5_1';
	  relu5_1name   ='relu5_1';
	  conv5_2name   ='conv5_2';
	  relu5_2name   ='relu5_2';
	  conv5_3name   ='conv5_3';
	  relu5_3name   ='relu5_3';
	  pool5_name    ='pool5';
	  fc6_name      ='fc6';
	  relu6_name    ='relu6'
	  drop6_name    ='drop6';
	  fc7_name      ='fc7';
	  relu6_name    ='relu7'
	  drop7_name    ='drop7';
	  
	else
	  bottom_data=['data_' num2str(i-1) 'p'];
	  conv1_1name   =['conv1_1' '_' num2str(i-1) 'p'];
	  relu1_1name     =['relu1_1' '_' num2str(i-1) 'p'];
	  conv1_2name   =['conv1_2' '_' num2str(i-1) 'p'];
	  relu1_2name     =['relu1_2' '_' num2str(i-1) 'p'];
	  pool1_name    =['pool1' '_' num2str(i-1) 'p'];
	  
	  conv2_1name   =['conv2_1' '_' num2str(i-1) 'p'];
	  relu2_1name   =['relu2_1' '_' num2str(i-1) 'p'];
	  conv2_2name   =['conv2_2' '_' num2str(i-1) 'p'];
	  relu2_2name   =['relu2_2' '_' num2str(i-1) 'p'];
	  pool2_name    =['pool2' '_' num2str(i-1) 'p'];
	  conv3_1name   =['conv3_1' '_' num2str(i-1) 'p'];
	  relu3_1name   =['relu3_1' '_' num2str(i-1) 'p'];
	  conv3_2name   =['conv3_2' '_' num2str(i-1) 'p'];
	  relu3_2name   =['relu3_2' '_' num2str(i-1) 'p'];
	  conv3_3name   =['conv3_3' '_' num2str(i-1) 'p'];
	  relu3_3name   =['relu3_3' '_' num2str(i-1) 'p'];
	  pool3_name    =['pool3' '_' num2str(i-1) 'p'];
	  conv4_1name   =['conv4_1' '_' num2str(i-1) 'p'];
	  relu4_1name   =['relu4_1' '_' num2str(i-1) 'p'];
	  conv4_2name   =['conv4_2' '_' num2str(i-1) 'p'];
	  relu4_2name   =['relu4_2' '_' num2str(i-1) 'p'];
	  conv4_3name   =['conv4_3' '_' num2str(i-1) 'p'];
	  relu4_3name   =['relu4_3' '_' num2str(i-1) 'p'];
	  pool4_name    =['pool4' '_' num2str(i-1) 'p'];
	  conv5_1name   =['conv5_1' '_' num2str(i-1) 'p'];
	  relu5_1name   =['relu5_1' '_' num2str(i-1) 'p'];
	  conv5_2name   =['conv5_2' '_' num2str(i-1) 'p'];
	  relu5_2name   =['relu5_2' '_' num2str(i-1) 'p'];
	  conv5_3name   =['conv5_3' '_' num2str(i-1) 'p'];
	  relu5_3name   =['relu5_3' '_' num2str(i-1) 'p'];
	  pool5_name    =['pool5' '_' num2str(i-1) 'p'];
	  
	  fc6_name      =['fc6' '_' num2str(i-1) 'p'];
	  relu6_name    =['relu6' '_' num2str(i-1) 'p'];
	  drop6_name    =['drop6' '_' num2str(i-1) 'p'];
	  fc7_name      =['fc7' '_' num2str(i-1) 'p'];
	  relu6_name    =['relu7' '_' num2str(i-1) 'p'];
	  drop7_name    =['drop7' '_' num2str(i-1) 'p'];
	end
	% layer conv1_1
	fprintf(fileID, 'layers %s\n','{');
	fprintf(fileID, 'bottom: "%s"\n',bottom_data);
	fprintf(fileID, 'top: "%s"\n',conv1_1name);
	fprintf(fileID, 'name: "%s"\n',conv1_1name);
	fprintf(fileID, 'type: %s\n','CONVOLUTION');
	fprintf(fileID, 'blobs_lr: %d\n',1);
	fprintf(fileID, 'blobs_lr: %d\n',2);
    fprintf(fileID, 'weight_decay: %d\n',1);
	fprintf(fileID, 'weight_decay: %d\n',0);
	fprintf(fileID, 'convolution_param  %s\n','{');
	fprintf(fileID, '  num_output:  %d\n',64);
	fprintf(fileID, '  pad:  %d\n',1);
	fprintf(fileID, '  kernel_size:  %d\n',3);
	fprintf(fileID, ' %s\n','}');
	fprintf(fileID, '%s: "conv1_1_w"\n','param');
	fprintf(fileID, '%s: "conv1_1_b"\n','param');
	fprintf(fileID, '%s\n\n','}')
	
 % layer relu_1_1
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'bottom: "%s"\n',conv1_1name);
  fprintf(fileID,'top: "%s"\n',conv1_1name);
  fprintf(fileID,'name: "%s"\n', relu1_1name);
  fprintf(fileID,'type: %s\n','RELU');
  fprintf(fileID, '%s\n\n','}')
    
  %layer conv1_2
  
  fprintf(fileID, 'layers %s\n','{');
	fprintf(fileID, 'bottom: "%s"\n',conv1_1name);
	fprintf(fileID, 'top: "%s"\n',conv1_2name);
	fprintf(fileID, 'name: "%s"\n',conv1_2name);
	fprintf(fileID, 'type: %s\n','CONVOLUTION');
	fprintf(fileID, 'blobs_lr: %d\n',1);
	fprintf(fileID, 'blobs_lr: %d\n',2);
    fprintf(fileID, 'weight_decay: %d\n',1);
	fprintf(fileID, 'weight_decay: %d\n',0);
	fprintf(fileID, 'convolution_param  %s\n','{');
	fprintf(fileID, '  num_output:  %d\n',64);
	fprintf(fileID, '  pad:  %d\n',1);
	fprintf(fileID, '  kernel_size:  %d\n',3);
	fprintf(fileID, ' %s\n','}');
	fprintf(fileID, '%s\n','param : "conv1_2_w"');
	fprintf(fileID, '%s\n', 'param: "conv1_2_b"');
	fprintf(fileID, '%s\n\n','}')
	

	
% layer relu_1_2
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'bottom: "%s"\n',conv1_2name);
  fprintf(fileID,'top: "%s"\n',conv1_2name);
  fprintf(fileID,'name: "%s"\n', relu1_2name);
  fprintf(fileID,'type: %s\n','RELU');
  fprintf(fileID, '%s\n\n','}')
  

	
	%layer pool1
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'bottom: "%s"\n',conv1_2name);
  fprintf(fileID,'top: "%s"\n',pool1_name);
  fprintf(fileID,'name: "%s"\n', pool1_name);
  fprintf(fileID,'type: %s\n','POOLING');
  fprintf(fileID, 'pooling_param  %s\n','{');
  fprintf(fileID, 'pool:  %s\n','MAX');
  fprintf(fileID, 'kernel_size:  %d\n',2);
  fprintf(fileID, 'stride:  %d\n',2);
  fprintf(fileID, ' %s\n','}');
  fprintf(fileID, '%s\n\n','}');
  
  
  
  %layer conv2_1
  
  fprintf(fileID, 'layers %s\n','{');
	fprintf(fileID, 'bottom: "%s"\n',pool1_name);
	fprintf(fileID, 'top: "%s"\n',conv2_1name);
	fprintf(fileID, 'name: "%s"\n',conv2_1name);
	fprintf(fileID, 'type: %s\n','CONVOLUTION');
	fprintf(fileID, 'blobs_lr: %d\n',1);
	fprintf(fileID, 'blobs_lr: %d\n',2);
    fprintf(fileID, 'weight_decay: %d\n',1);
	fprintf(fileID, 'weight_decay: %d\n',0);
	fprintf(fileID, 'convolution_param  %s\n','{');
	fprintf(fileID, '  num_output:  %d\n',128);
	fprintf(fileID, '  pad:  %d\n',1);
	fprintf(fileID, '  kernel_size:  %d\n',3);
	fprintf(fileID, ' %s\n','}');
	fprintf(fileID, '%s: "conv2_1_w"\n','param');
	fprintf(fileID, '%s: "conv2_1_b"\n','param');
	fprintf(fileID, '%s\n\n','}')
	
	
	% layer relu_2_1
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'bottom: "%s"\n',conv2_1name);
  fprintf(fileID,'top: "%s"\n',conv2_1name);
  fprintf(fileID,'name: "%s"\n', relu2_1name);
  fprintf(fileID,'type: %s\n','RELU');
  fprintf(fileID, '%s\n\n','}');
  
  
   %layer conv2_2
  
  fprintf(fileID, 'layers %s\n','{');
	fprintf(fileID, 'bottom: "%s"\n',conv2_1name);
	fprintf(fileID, 'top: "%s"\n',conv2_2name);
	fprintf(fileID, 'name: "%s"\n',conv2_2name);
	fprintf(fileID, 'type: %s\n','CONVOLUTION');
	fprintf(fileID, 'blobs_lr: %d\n',1);
	fprintf(fileID, 'blobs_lr: %d\n',2);
    fprintf(fileID, 'weight_decay: %d\n',1);
	fprintf(fileID, 'weight_decay: %d\n',0);
	fprintf(fileID, 'convolution_param  %s\n','{');
	fprintf(fileID, '  num_output:  %d\n',128);
	fprintf(fileID, '  pad:  %d\n',1);
	fprintf(fileID, '  kernel_size:  %d\n',3);
	fprintf(fileID, ' %s\n','}');
	fprintf(fileID, '%s: "conv2_2_w"\n','param ');
	fprintf(fileID, '%s: "conv2_2_b"\n','param ');
	fprintf(fileID, '%s\n\n','}');
	
	
	
		% layer relu_2_2
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'bottom: "%s"\n',conv2_2name);
  fprintf(fileID,'top: "%s"\n',conv2_2name);
  fprintf(fileID,'name: "%s"\n', relu2_2name);
  fprintf(fileID,'type: %s\n','RELU');
  fprintf(fileID, '%s\n\n','}');
  
  
  %layer pool2
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'bottom: "%s"\n',conv2_2name);
  fprintf(fileID,'top: "%s"\n',pool2_name);
  fprintf(fileID,'name: "%s"\n', pool2_name);
  fprintf(fileID,'type: %s\n','POOLING');
  fprintf(fileID, 'pooling_param  %s\n','{');
  fprintf(fileID, 'pool:  %s\n','MAX');
  fprintf(fileID, 'kernel_size:  %d\n',2);
  fprintf(fileID, 'stride:  %d\n',2);
  fprintf(fileID, ' %s\n','}');
  fprintf(fileID, '%s\n\n','}');
  
  
  %layer conv3_1
  
  fprintf(fileID, 'layers %s\n','{');
	fprintf(fileID, 'bottom: "%s"\n',pool2_name);
	fprintf(fileID, 'top: "%s"\n',conv3_1name);
	fprintf(fileID, 'name: "%s"\n',conv3_1name);
	fprintf(fileID, 'type: %s\n','CONVOLUTION');
	fprintf(fileID, 'blobs_lr: %d\n',1);
	fprintf(fileID, 'blobs_lr: %d\n',2);
    fprintf(fileID, 'weight_decay: %d\n',1);
	fprintf(fileID, 'weight_decay: %d\n',0);
	fprintf(fileID, 'convolution_param  %s\n','{');
	fprintf(fileID, '  num_output:  %d\n',256);
	fprintf(fileID, '  pad:  %d\n',1);
	fprintf(fileID, '  kernel_size:  %d\n',3);
	fprintf(fileID, ' %s\n','}');
	fprintf(fileID, '%s: "conv3_1_w"\n','param ');
	fprintf(fileID, '%s: "conv3_1_b"\n','param ');
	fprintf(fileID, '%s\n\n','}');
	
	
	
		% layer relu3_1
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'bottom: "%s"\n',conv3_1name);
  fprintf(fileID,'top: "%s"\n',conv3_1name);
  fprintf(fileID,'name: "%s"\n', relu3_1name);
  fprintf(fileID,'type: %s\n','RELU');
  fprintf(fileID, '%s\n\n','}');
  
  
  %layer conv3_2
  
    fprintf(fileID, 'layers %s\n','{');
	fprintf(fileID, 'bottom: "%s"\n',conv3_1name);
	fprintf(fileID, 'top: "%s"\n',conv3_2name);
	fprintf(fileID, 'name: "%s"\n',conv3_2name);
	fprintf(fileID, 'type: %s\n','CONVOLUTION');
	fprintf(fileID, 'blobs_lr: %d\n',1);
	fprintf(fileID, 'blobs_lr: %d\n',2);
    fprintf(fileID, 'weight_decay: %d\n',1);
	fprintf(fileID, 'weight_decay: %d\n',0);
	fprintf(fileID, 'convolution_param  %s\n','{');
	fprintf(fileID, '  num_output:  %d\n',256);
	fprintf(fileID, '  pad:  %d\n',1);
	fprintf(fileID, '  kernel_size:  %d\n',3);
	fprintf(fileID, ' %s\n','}');
	fprintf(fileID, '%s: "conv3_2_w"\n','param ');
	fprintf(fileID, '%s: "conv3_2_b"\n','param ');
	fprintf(fileID, '%s\n\n','}');
	
		% layer relu3_2
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'bottom: "%s"\n',conv3_2name);
  fprintf(fileID,'top: "%s"\n',conv3_2name);
  fprintf(fileID,'name: "%s"\n', relu3_2name);
  fprintf(fileID,'type: %s\n','RELU');
  fprintf(fileID, '%s\n\n','}');
  
   %layer conv3_3
  
    fprintf(fileID, 'layers %s\n','{');
	fprintf(fileID, 'bottom: "%s"\n',conv3_2name);
	fprintf(fileID, 'top: "%s"\n',conv3_3name);
	fprintf(fileID, 'name: "%s"\n',conv3_3name);
	fprintf(fileID, 'type: %s\n','CONVOLUTION');
	fprintf(fileID, 'blobs_lr: %d\n',1);
	fprintf(fileID, 'blobs_lr: %d\n',2);
    fprintf(fileID, 'weight_decay: %d\n',1);
	fprintf(fileID, 'weight_decay: %d\n',0);
	fprintf(fileID, 'convolution_param  %s\n','{');
	fprintf(fileID, '  num_output:  %d\n',256);
	fprintf(fileID, '  pad:  %d\n',1);
	fprintf(fileID, '  kernel_size:  %d\n',3);
	fprintf(fileID, ' %s\n','}');
	fprintf(fileID, '%s: "conv3_3_w"\n','param ');
	fprintf(fileID, '%s: "conv3_3_b"\n','param ');
	fprintf(fileID, '%s\n\n','}');
	
		% layer relu3_3
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'bottom: "%s"\n',conv3_3name);
  fprintf(fileID,'top: "%s"\n',conv3_3name);
  fprintf(fileID,'name: "%s"\n', relu3_3name);
  fprintf(fileID,'type: %s\n','RELU');
  fprintf(fileID, '%s\n\n','}');
  
  
  %layer pool3
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'bottom: "%s"\n',conv3_3name);
  fprintf(fileID,'top: "%s"\n',pool3_name);
  fprintf(fileID,'name: "%s"\n', pool3_name);
  fprintf(fileID,'type: %s\n','POOLING');
  fprintf(fileID, 'pooling_param  %s\n','{');
  fprintf(fileID, 'pool:  %s\n','MAX');
  fprintf(fileID, 'kernel_size:  %d\n',2);
  fprintf(fileID, 'stride:  %d\n',2);
  fprintf(fileID, ' %s\n','}');
  fprintf(fileID, '%s\n\n','}');
  
  
  
  
  
  
  %layer conv4_1
  
  fprintf(fileID, 'layers %s\n','{');
	fprintf(fileID, 'bottom: "%s"\n',pool3_name);
	fprintf(fileID, 'top: "%s"\n',conv4_1name);
	fprintf(fileID, 'name: "%s"\n',conv4_1name);
	fprintf(fileID, 'type: %s\n','CONVOLUTION');
	fprintf(fileID, 'blobs_lr: %d\n',1);
	fprintf(fileID, 'blobs_lr: %d\n',2);
    fprintf(fileID, 'weight_decay: %d\n',1);
	fprintf(fileID, 'weight_decay: %d\n',0);
	fprintf(fileID, 'convolution_param  %s\n','{');
	fprintf(fileID, '  num_output:  %d\n',512);
	fprintf(fileID, '  pad:  %d\n',1);
	fprintf(fileID, '  kernel_size:  %d\n',3);
	fprintf(fileID, ' %s\n','}');
	fprintf(fileID, '%s: "conv4_1_w"\n','param ');
	fprintf(fileID, '%s: "conv4_1_b"\n','param ');
	fprintf(fileID, '%s\n\n','}');
	
	
	
  % layer relu4_1
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'bottom: "%s"\n',conv4_1name);
  fprintf(fileID,'top: "%s"\n',conv4_1name);
  fprintf(fileID,'name: "%s"\n', relu4_1name);
  fprintf(fileID,'type: %s\n','RELU');
  fprintf(fileID, '%s\n\n','}');
  
    %layer conv4_2
  fprintf(fileID, 'layers %s\n','{');
	fprintf(fileID, 'bottom: "%s"\n',conv4_1name);
	fprintf(fileID, 'top: "%s"\n',conv4_2name);
	fprintf(fileID, 'name: "%s"\n',conv4_2name);
	fprintf(fileID, 'type: %s\n','CONVOLUTION');
	fprintf(fileID, 'blobs_lr: %d\n',1);
	fprintf(fileID, 'blobs_lr: %d\n',2);
    fprintf(fileID, 'weight_decay: %d\n',1);
	fprintf(fileID, 'weight_decay: %d\n',0);
	fprintf(fileID, 'convolution_param  %s\n','{');
	fprintf(fileID, '  num_output:  %d\n',512);
	fprintf(fileID, '  pad:  %d\n',1);
	fprintf(fileID, '  kernel_size:  %d\n',3);
	fprintf(fileID, ' %s\n','}');
	fprintf(fileID, '%s: "conv4_2_w"\n','param ');
	fprintf(fileID, '%s: "conv4_2_b"\n','param ');
	fprintf(fileID, '%s\n\n','}');
	
	
	% layer relu4_2
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'bottom: "%s"\n',conv4_2name);
  fprintf(fileID,'top: "%s"\n',conv4_2name);
  fprintf(fileID,'name: "%s"\n', relu4_2name);
  fprintf(fileID,'type: %s\n','RELU');
  fprintf(fileID, '%s\n\n','}');
  
  
   %layer conv4_3
  fprintf(fileID, 'layers %s\n','{');
	fprintf(fileID, 'bottom: "%s"\n',conv4_2name);
	fprintf(fileID, 'top: "%s"\n',conv4_3name);
	fprintf(fileID, 'name: "%s"\n',conv4_3name);
	fprintf(fileID, 'type: %s\n','CONVOLUTION');
	fprintf(fileID, 'blobs_lr: %d\n',1);
	fprintf(fileID, 'blobs_lr: %d\n',2);
    fprintf(fileID, 'weight_decay: %d\n',1);
	fprintf(fileID, 'weight_decay: %d\n',0);
	fprintf(fileID, 'convolution_param  %s\n','{');
	fprintf(fileID, '  num_output:  %d\n',512);
	fprintf(fileID, '  pad:  %d\n',1);
	fprintf(fileID, '  kernel_size:  %d\n',3);
	fprintf(fileID, ' %s\n','}');
	fprintf(fileID, '%s: "conv4_3_w"\n','param ');
	fprintf(fileID, '%s: "conv4_3_b"\n','param ');
	fprintf(fileID, '%s\n\n','}');
	
	
	% layer relu4_3
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'bottom: "%s"\n',conv4_3name);
  fprintf(fileID,'top: "%s"\n',conv4_3name);
  fprintf(fileID,'name: "%s"\n', relu4_3name);
  fprintf(fileID,'type: %s\n','RELU');
  fprintf(fileID, '%s\n\n','}');
  
  
  %layer pool4
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'bottom: "%s"\n',conv4_3name);
  fprintf(fileID,'top: "%s"\n',pool4_name);
  fprintf(fileID,'name: "%s"\n', pool4_name);
  fprintf(fileID,'type: %s\n','POOLING');
  fprintf(fileID, 'pooling_param  %s\n','{');
  fprintf(fileID, 'pool:  %s\n','MAX');
  fprintf(fileID, 'kernel_size:  %d\n',2);
  fprintf(fileID, 'stride:  %d\n',2);
  fprintf(fileID, ' %s\n','}');
  fprintf(fileID, '%s\n\n','}');
  
  
  %layer conv5_1
  
  fprintf(fileID, 'layers %s\n','{');
	fprintf(fileID, 'bottom: "%s"\n',pool4_name);
	fprintf(fileID, 'top: "%s"\n',conv5_1name);
	fprintf(fileID, 'name: "%s"\n',conv5_1name);
	fprintf(fileID, 'type: %s\n','CONVOLUTION');
	fprintf(fileID, 'blobs_lr: %d\n',1);
	fprintf(fileID, 'blobs_lr: %d\n',2);
    fprintf(fileID, 'weight_decay: %d\n',1);
	fprintf(fileID, 'weight_decay: %d\n',0);
	fprintf(fileID, 'convolution_param  %s\n','{');
	fprintf(fileID, '  num_output:  %d\n',512);
	fprintf(fileID, '  pad:  %d\n',1);
	fprintf(fileID, '  kernel_size:  %d\n',3);
	fprintf(fileID, ' %s\n','}');
	fprintf(fileID, '%s: "conv5_1_w"\n','param ');
	fprintf(fileID, '%s: "conv5_1_b"\n','param ');
	fprintf(fileID, '%s\n\n','}');
	
	% layer relu5_1
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'bottom: "%s"\n',conv5_1name);
  fprintf(fileID,'top: "%s"\n',conv5_1name);
  fprintf(fileID,'name: "%s"\n', relu5_1name);
  fprintf(fileID,'type: %s\n','RELU');
  fprintf(fileID, '%s\n\n','}');
  
  
  %layer conv5_2
  
  fprintf(fileID, 'layers %s\n','{');
	fprintf(fileID, 'bottom: "%s"\n',conv5_1name);
	fprintf(fileID, 'top: "%s"\n',conv5_2name);
	fprintf(fileID, 'name: "%s"\n',conv5_2name);
	fprintf(fileID, 'type: %s\n','CONVOLUTION');
	fprintf(fileID, 'blobs_lr: %d\n',1);
	fprintf(fileID, 'blobs_lr: %d\n',2);
    fprintf(fileID, 'weight_decay: %d\n',1);
	fprintf(fileID, 'weight_decay: %d\n',0);
	fprintf(fileID, 'convolution_param  %s\n','{');
	fprintf(fileID, '  num_output:  %d\n',512);
	fprintf(fileID, '  pad:  %d\n',1);
	fprintf(fileID, '  kernel_size:  %d\n',3);
	fprintf(fileID, ' %s\n','}');
	fprintf(fileID, '%s: "conv5_2_w"\n','param ');
	fprintf(fileID, '%s: "conv5_2_b"\n','param ');
	fprintf(fileID, '%s\n\n','}');
	
	% layer relu5_1
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'bottom: "%s"\n',conv5_2name);
  fprintf(fileID,'top: "%s"\n',conv5_2name);
  fprintf(fileID,'name: "%s"\n', relu5_2name);
  fprintf(fileID,'type: %s\n','RELU');
  fprintf(fileID, '%s\n\n','}');
  
  
  
  
   %layer conv5_3
  
  fprintf(fileID, 'layers %s\n','{');
	fprintf(fileID, 'bottom: "%s"\n',conv5_2name);
	fprintf(fileID, 'top: "%s"\n',conv5_3name);
	fprintf(fileID, 'name: "%s"\n',conv5_3name);
	fprintf(fileID, 'type: %s\n','CONVOLUTION');
	fprintf(fileID, 'blobs_lr: %d\n',1);
	fprintf(fileID, 'blobs_lr: %d\n',2);
    fprintf(fileID, 'weight_decay: %d\n',1);
	fprintf(fileID, 'weight_decay: %d\n',0);
	fprintf(fileID, 'convolution_param  %s\n','{');
	fprintf(fileID, '  num_output:  %d\n',512);
	fprintf(fileID, '  pad:  %d\n',1);
	fprintf(fileID, '  kernel_size:  %d\n',3);
	fprintf(fileID, ' %s\n','}');
	fprintf(fileID, '%s: "conv5_3_w"\n','param ');
	fprintf(fileID, '%s: "conv5_3_b"\n','param ');
	fprintf(fileID, '%s\n\n','}');
	
	% layer relu5_3
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'bottom: "%s"\n',conv5_3name);
  fprintf(fileID,'top: "%s"\n',conv5_3name);
  fprintf(fileID,'name: "%s"\n', relu5_3name);
  fprintf(fileID,'type: %s\n','RELU');
  fprintf(fileID, '%s\n\n','}');
  
  
  %layer pool5
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'bottom: "%s"\n',conv5_3name);
  fprintf(fileID,'top: "%s"\n',pool5_name);
  fprintf(fileID,'name: "%s"\n', pool5_name);
  fprintf(fileID,'type: %s\n','POOLING');
  fprintf(fileID, 'pooling_param  %s\n','{');
  fprintf(fileID, 'pool:  %s\n','MAX');
  fprintf(fileID, 'kernel_size:  %d\n',2);
  fprintf(fileID, 'stride:  %d\n',2);
  fprintf(fileID, ' %s\n','}');
  fprintf(fileID, '%s\n\n','}');
  
  %layer fc6
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'bottom: "%s"\n',pool5_name);
  fprintf(fileID,'top: "%s"\n',fc6_name);
  fprintf(fileID,'name: "%s"\n', fc6_name);
  fprintf(fileID,'type: %s\n','INNER_PRODUCT');
  fprintf(fileID, 'inner_product_param  %s\n','{');
  fprintf(fileID, 'num_output:  %d\n',4096);
  fprintf(fileID, ' %s\n','}');
  fprintf(fileID, '%s: "fc6_w"\n','param');
  fprintf(fileID, '%s: "fc6_b"\n','param');
  fprintf(fileID, '%s\n\n','}');
  
  % layer relu6
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'bottom: "%s"\n',fc6_name);
  fprintf(fileID,'top: "%s"\n',fc6_name);
  fprintf(fileID,'name: "%s"\n', relu6_name);
  fprintf(fileID,'type: %s\n','RELU');
  fprintf(fileID, '%s\n\n','}');
  
   % layer drop6
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'bottom: "%s"\n',fc6_name);
  fprintf(fileID,'top: "%s"\n',fc6_name);
  fprintf(fileID,'name: "%s"\n', drop6_name);
  fprintf(fileID,'type: %s\n','DROPOUT');
  fprintf(fileID, 'dropout_param  %s\n','{');
  fprintf(fileID, 'dropout_ratio:  %.1f\n',0.5);
  fprintf(fileID, ' %s\n','}')
  fprintf(fileID, '%s\n\n','}');
  
  %layer fc7
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'bottom: "%s"\n',fc6_name);
  fprintf(fileID,'top: "%s"\n',fc7_name);
  fprintf(fileID,'name: "%s"\n', fc7_name);
  fprintf(fileID,'type: %s\n','INNER_PRODUCT');
  fprintf(fileID, 'inner_product_param  %s\n','{');
  fprintf(fileID, 'num_output:  %d\n',4096);
  fprintf(fileID, ' %s\n','}');
  fprintf(fileID, '%s: "fc7_w"\n','param ');
  fprintf(fileID, '%s: "fc7_b"\n','param ');
  fprintf(fileID, '%s\n\n','}');
  
  
   % layer drop7
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'bottom: "%s"\n',fc7_name);
  fprintf(fileID,'top: "%s"\n',fc7_name);
  fprintf(fileID,'name: "%s"\n', drop7_name);
  fprintf(fileID,'type: %s\n','DROPOUT');
  fprintf(fileID, 'dropout_param  %s\n','{');
  fprintf(fileID, 'dropout_ratio:  %.1f\n',0.5);
  fprintf(fileID, ' %s\n','}')
  fprintf(fileID, '%s\n\n','}');
  
end



 % layer eltmax_layer
fprintf(fileID,'layers %s\n','{');
fc7_bottom='fc7'
fprintf(fileID,'bottom: "%s"\n',fc7_bottom);
for i=2:num_of_shared_net
	fc7_bottom_extra=[fc7_bottom '_' num2str(i-1) 'p'];
	fprintf(fileID,'bottom: "%s"\n',fc7_bottom_extra);
end
	fprintf(fileID,'top: "%s"\n','fc7_eltmax');
	fprintf(fileID,'name: "%s"\n', 'eltmax_layer');
	fprintf(fileID,'type: %s\n','ELTWISE');
	fprintf(fileID, 'eltwise_param  %s\n','{');
	fprintf(fileID, ' operation:  %s\n','MAX');
	fprintf(fileID, ' %s\n','}');
	fprintf(fileID, '%s\n\n','}');

%layer fc8

%layer fc7
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'bottom: "%s"\n','fc7_eltmax');
  fprintf(fileID,'top: "%s"\n','fc_gene_ontology');
  fprintf(fileID,'name: "%s"\n', 'fc_gene_ontology');
  fprintf(fileID,'type: %s\n','INNER_PRODUCT');
  
  fprintf(fileID, 'blobs_lr: %d\n',10);
  fprintf(fileID, 'blobs_lr: %d\n',20);
  fprintf(fileID, 'weight_decay: %d\n',1);
  fprintf(fileID, 'weight_decay: %d\n',0);
  fprintf(fileID, 'inner_product_param  %s\n','{');
  fprintf(fileID, 'num_output:  %d\n',81);
  
  fprintf(fileID, '  weight_filler:  %s\n','{');
  fprintf(fileID, '  type: "%s"\n','gaussian');
  fprintf(fileID, '  std: %.2f\n',0.01);
  fprintf(fileID, '  %s\n','}');
  
  fprintf(fileID, '   bias_filler:  %s\n','{');
  fprintf(fileID, '  type: "%s"\n','constant');
  fprintf(fileID, '  value: %d\n',0);
  fprintf(fileID, '  %s\n','}');
  
   fprintf(fileID, '%s\n','}');
  fprintf(fileID, '%s\n\n','}');

  %layer loss
  fprintf(fileID,'layers %s\n','{');
  fprintf(fileID,'name: "%s"\n', 'loss');
  fprintf(fileID,'type: %s\n','MULTI_LABEL_LOSS');
  fprintf(fileID,'bottom: "%s"\n','fc_gene_ontology');
  fprintf(fileID,'bottom: "%s"\n','label');
  fprintf(fileID, '%s\n\n','}');
 
end