function images = prepare_vgg_image_No_crop(imdata)
% ------------------------------------------------------------------------
rgb_dim=3;
mean_pix = [103.939, 116.779, 123.68];
IMAGE_DIM = 224;
%CROPPED_DIM = 224;
num_rgb_chs=3;
if strcmp(class(imdata),'cell')
   num_img = length(imdata);
   images = zeros(IMAGE_DIM, IMAGE_DIM, num_rgb_chs*num_img, 1, 'single');
   for i=1:num_img 
        im=imdata{i};
		% resize to fixed input size
		im = single(im);

		im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');

		% RGB -> BGR
		im = im(:, :, [3 2 1]);
		im = permute(im, [2 1 3]);
		% oversample (4 corners, center, and their x-axis flips)
		
        cur_rgb_ch_idx= (i-1)*num_rgb_chs+1;
		images(:,:,cur_rgb_ch_idx:cur_rgb_ch_idx+2,1)=im;
		%images(:,:,:,1) = im;

		% mean BGR pixel subtraction
		%for c = 1:3
		%cur_rgb_ch_idx:cur_rgb_ch_idx+2
		for c =cur_rgb_ch_idx:cur_rgb_ch_idx+2
			images(:, :, c, :) = images(:, :, c, :)-mean_pix(mod(c-1,3)+1);
		end
	end
	%disp('images size in prepare_vgg_image_No_crop is:')
	%size(images)
else
   images = zeros(IMAGE_DIM, IMAGE_DIM, num_rgb_chs, 1, 'single');
        im=imdata;
		% resize to fixed input size
		im = single(im);

		im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');

		% RGB -> BGR
		im = im(:, :, [3 2 1]);
		im = permute(im, [2 1 3]);
		% oversample (4 corners, center, and their x-axis flips)
		
       % cur_rgb_ch_idx= (i-1)*num_rgb_chs+1;
		images(:,:,:,1) = im;

		% mean BGR pixel subtraction
		for c = 1:3
		images(:, :, c, :) = images(:, :, c, :)-mean_pix(c);
		end
	end

end


