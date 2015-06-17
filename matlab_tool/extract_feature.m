function feature =extract_feature(net,imageFile, layer_name)
    
	
	if nargin<3
		layer_name ='conv4_2';
	end
	
	if strcmp(class(imageFile),'cell')
		for i=1:length(imageFile)
			im{i}=imread(imageFile{i});
		end
    else
	    im=imread(imageFile);
	end
	%disp(['numof images is =' num2str(length(im))]);
	input_data = {prepare_vgg_image_No_crop(im)};
	%disp('size of input is :' );
	%size(length(input_data))
	scores = net.forward(input_data);
	
	feature =net.get_blob_data(layer_name); %layer 12/(13, if layer start from 1)
	 
	
    %d=net.get_blob_data('pool_4') ;% layer 14
    %feature =feature (:);
end