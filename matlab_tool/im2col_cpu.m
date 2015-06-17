function [dc_idx, dm_idx] = im2col_cpu(data_im, channels, height, width, ksize, pad, stride)
  height_col = (height + 2 * pad - ksize) / stride + 1;
  width_col = (width + 2 * pad - ksize) / stride + 1;
  channels_col = channels * ksize * ksize;
  count=1;
  for c = 0:channels_col-1
    w_offset = mod(c, ksize); % ksize;
    h_offset = mod( floor(c / ksize),ksize); % ksize;
    c_im =floor( c / (ksize* ksize));
    
    for h = 0:height_col-1
      for  w = 0:width_col-1
           h_pad = h * stride - pad + h_offset;
           w_pad = w * stride - pad + w_offset;
           
        if  h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width
            d_col_idx =(c * height_col + h) * width_col + w;
            d_im_idx  = (c_im * height + h_pad) * width + w_pad;
            dc_idx(count)=d_col_idx;
            dm_idx(count)=d_im_idx;
            
            %data_col(d_col_idx+1) = data_im(d_im_idx+1);
        else
           dc_idx(count)=(c * height_col + h) * width_col + w;
           dm_idx(count)=-1;
           %data_col((c * height_col + h) * width_col + w+1) = 0;
         % count=count+1;
          %x=1;
        end
         count =count+1;
      end
    end
    x=1;
  end
end
