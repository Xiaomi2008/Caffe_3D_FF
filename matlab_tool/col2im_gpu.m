function [data_im,  da_all_idx]= col2im_gpu(data_col, channels, height, width, depth, ksize,  pad, stride) 
  height_col = floor((height + 2 * pad - ksize) / stride) + 1;
  width_col = floor((width + 2 * pad - ksize) / stride) + 1;
  depth_col = floor((depth + 2 * pad - ksize) / stride) + 1;
  num_kernels = channels * height * width * depth;
 
  [data_im,  da_all_idx]=col2im_gpu_kernel( num_kernels, data_col, height, width, depth, channels, ksize, pad, stride,  height_col, width_col, depth_col);
end

function [data_im,  da_all_idx]=col2im_gpu_kernel(n, data_col, height, width, depth, channels, ksize,pad, stride, height_col,width_col,  depth_col)
    
    for index=0:n-1
           val = 0;
	 d = mod(index, depth) + pad;
     w = mod(floor(index /depth) , width) + pad;
     h = mod(floor(index / (depth * width)), height) + pad;
     c = floor(index / (depth * width * height ));
    %compute the start and end of the output
	if d < ksize
		d_col_start =0;
	else
	    d_col_start   = floor((d - ksize) / stride) + 1;
	end
	d_col_end = min(floor(d / stride) + 1, depth_col);
	
	if w < ksize
		w_col_start =0;
	else
	    w_col_start   = floor((w - ksize) / stride) + 1;
	end
	w_col_end = min(floor(w / stride) + 1, width_col);
	
	if h < ksize
		h_col_start =0;
	else
	    h_col_start   = floor((h - ksize) / stride) + 1;
	end
	h_col_end = min(floor(h / stride) + 1, height_col);
	
	% d_col_start = (d < ksize) ? 0 : (d - ksize) / stride + 1;
     
     %w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
     %w_col_end = min(w / stride + 1, width_col);
    % h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
     %h_col_end = min(h / stride + 1, height_col);
     k3=ksize * ksize *ksize;
	 k2=ksize * ksize;
     count =1;
     d_idx=0;
     p_dix=0;
     for h_col = h_col_start: h_col_end-1
      for w_col = w_col_start: w_col_end-1
	    for d_col = d_col_start: d_col_end-1
			%// the col location: [c * width * height + h_out, w_out]
			%//int c_col = c * ksize * ksize + (h - h_col * stride) * ksize + (w - w_col * stride);
			%//int c_col = c *  + (h - h_col * stride) * ksize *ksize + (w - w_col * stride)*ksize + (d - d_col * stride);
			c_col = c * k3 + (h - h_col * stride) * k2 + (w - w_col * stride)*ksize + (d - d_col * stride);
			d_idx(count)=((c_col * height_col + h_col) * width_col + w_col) * depth_col +d_col;
             p_idx(count).h=h_col;
             p_idx(count).w=w_col;
             p_idx(count).d=d_col;
			val = val + data_col(d_idx(count)+1);
            count=count+1;
		end
      end
     end
     da_all_idx(index+1).h=h;
     da_all_idx(index+1).w=w;
     da_all_idx(index+1).d=d;
     da_all_idx(index+1).src_idx=d_idx;
     da_all_idx(index+1).h_start=h_col_start;
     da_all_idx(index+1).w_start=w_col_start;
     da_all_idx(index+1).d_start=d_col_start;
     
     da_all_idx(index+1).h_end=h_col_end;
     da_all_idx(index+1).w_end=w_col_end;
     da_all_idx(index+1).d_end=d_col_end;
     da_all_idx(index+1).p_idx =p_idx;
     
	 data_im(index+1) = val;
    end
end