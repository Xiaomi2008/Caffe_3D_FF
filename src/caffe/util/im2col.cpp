// Copyright 2014 BVLC and contributors.

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

//template <typename Dtype>
// void im2col_cpu(const Dtype* data_im, const int channels,
    // const int height, const int width, const int ksize, const int pad,
    // const int stride, Dtype* data_col) {
  // int height_col = (height + 2 * pad - ksize) / stride + 1;
  // int width_col = (width + 2 * pad - ksize) / stride + 1;
  // int channels_col = channels * ksize * ksize;
  // for (int c = 0; c < channels_col; ++c) {
    // int w_offset = c % ksize;
    // int h_offset = (c / ksize) % ksize;
    // int c_im = c / ksize / ksize;
    // for (int h = 0; h < height_col; ++h) {
      // for (int w = 0; w < width_col; ++w) {
        // int h_pad = h * stride - pad + h_offset;
        // int w_pad = w * stride - pad + w_offset;
        // if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          // data_col[(c * height_col + h) * width_col + w] =
            // data_im[(c_im * height + h_pad) * width + w_pad];
        // else
          // data_col[(c * height_col + h) * width_col + w] = 0;
      // }
    // }
  // }
// }


// For sparse Kernel operation - pixel wise labeling
template <typename Dtype>
 void im2col_sk_cpu(const Dtype* data_im, const int channels,
     const int height, const int width, const int depth,
	 const int kernel_h, const int kernel_w, const int kernel_d,
     const int pad_h, const int pad_w, const int pad_d,
     const int stride_h, const int stride_w, const int stride_d,
     const int kstride_h, const int kstride_w, const int kstride_d,
     Dtype* data_col) {
   
    size_t ext_kernel_h = (kernel_h - 1) * kstride_h + 1;
    size_t ext_kernel_w = (kernel_w - 1) * kstride_w + 1;
    size_t ext_kernel_d = (kernel_d - 1) * kstride_d + 1;
  
	size_t height_out 	= (height - ext_kernel_h) / stride_h + 1;
	size_t width_out  	= (width - ext_kernel_w) / stride_w + 1;
	size_t depth_out 	= (depth - ext_kernel_d) / stride_d + 1;
  
  
  LOG(INFO)<<channels<<" "<<height_out<<" "<<
          width_out <<" "<< depth_out<<" "<< kernel_h<<" "<< kernel_w<<" "<<kernel_d<<" "<< pad_h<<" "<< pad_w<<" "<< pad_d<<" "<< stride_h<<" "<< stride_w<<" "<<stride_d<<" "<<
          kstride_h<<" "<< kstride_w<<" "<< kstride_d;
  
  // LOG(INFO)<<
 // col_buffer_.Reshape(
  //    1, channels_ * kernel_h_ * kernel_w_ *kernel_d_, height_out, width_out, depth_out);
  
  size_t  size_col_buf =(size_t)channels * (size_t)kernel_h * (size_t)kernel_w *(size_t)kernel_d*(size_t)height_out*(size_t)width_out*(size_t)depth_out;
  
	
    //int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    //int width_col  = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    //int depth_col  = (depth + 2 * pad_d - kernel_d) / stride_d + 1;
	
	size_t height_col = ((size_t)height + 2 * (size_t)pad_h - ext_kernel_h) / (size_t)stride_h + 1;
    size_t width_col  = ((size_t)width + 2 * (size_t)pad_w - ext_kernel_w) / (size_t)stride_w + 1;
    size_t depth_col  = ((size_t)depth + 2 * (size_t)pad_d - ext_kernel_d) / (size_t)stride_d + 1;
	
    size_t channels_col = (size_t)channels * (size_t)kernel_h * (size_t)kernel_w * (size_t)kernel_d;
   // CHECK_GE(channels_col,1);
   //LOG(INFO)<< "channels_col = "<<channels_col <<"size_buf =" <<size_col_buf ;
    size_t t_kernel = (size_t)kernel_h * (size_t)kernel_w * (size_t)kernel_d;
   for (size_t c = 0; c < channels_col; ++c) {
    size_t d_offset = c% kernel_d;
    size_t w_offset = (c / kernel_d) % kernel_w;
    size_t h_offset = (c / kernel_d / kernel_w) % kernel_h;
    size_t c_im = c / t_kernel;
	
   
     for (size_t h = 0; h < height_col; ++h) {
	      size_t h_pad = h * stride_h - pad_h + h_offset*kstride_h;
       for (size_t w = 0; w < width_col; ++w) {
	   //for (int w = 0; w < end_w; ++w) {
	       size_t w_pad = w * stride_w - pad_w + w_offset*kstride_w;
		   for(size_t d = 0 ; d < depth_col ; ++d){
		        size_t   d_pad = d * stride_d - pad_d + d_offset*kstride_d;	
				size_t  idx_col=(size_t)(((c * height_col + h) * width_col + w) * depth_col +d);
				if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width && d_pad >= 0 && d_pad < depth)
				    {	
						size_t  idx_c_im=((c_im * height + h_pad) * width + w_pad) * depth + d_pad;
						//if (idx_col > size_col_buf )
						// {LOG(INFO)<<"idx is out of array!!" << " idx_col  = " << idx_col << "size of arary = " << size_col_buf <<" "<<c <<" "<< h<< " "<<w<<" "<<d;}
						 
						   //if(idx_col<0 ){LOG(INFO)<<"((c * height_col + h) * width_col + w) * depth_col +d;"<<" c"<<c<<" hc"<< height_col<<" h "<<h<<" width_col "<<width_col << " w " <<w<<" depth_col "<<depth_col<<" d "<<d;} 
						   data_col[idx_col] = data_im[idx_c_im];
						   //data_col[size_col_buf] = data_im[idx_c_im];
						//int x=0;
					}
				else
				  //data_col[((c * height_col + h) * width_col + w) * depth_col +d] = 0;
				  
				    //data_col[idx_col]  =0;
					data_col[idx_col]  =0;
			  
					// // int h_pad = h * stride_h - pad_h + h_offset * kstride_h;
					// // int w_pad = w * stride_w - pad_w + w_offset * kstride_w;
					 // if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
					   // data_col[(c * height_col + h) * width_col + w] =
						 // data_im[(c_im * height + h_pad) * width + w_pad];
					 // else
					   // data_col[(c * height_col + h) * width_col + w] = 0;
			}
       }
     }
   }
   //LOG(INFO)<<"don im2col_sk";
 }
 
 // Explicit instantiation
 template void im2col_sk_cpu<float>(const float* data_im, const int channels,
     const int height, const int width, const int depth,
	 const int kernel_h, const int kernel_w, const int kernel_d,
     const int pad_h, const int pad_w, const int pad_d,
     const int stride_h, const int stride_w, const int stride_d,
     const int kstride_h, const int kstride_w, const int kstride_d,
     float* data_col);
 template void im2col_sk_cpu<double>(const double* data_im, const int channels,
     const int height, const int width, const int depth,
	 const int kernel_h, const int kernel_w, const int kernel_d,
     const int pad_h, const int pad_w, const int pad_d,
     const int stride_h, const int stride_w, const int stride_d,
     const int kstride_h, const int kstride_w, const int kstride_d,
     double* data_col);
	 
	 
	
template <typename Dtype>
void im2col_sk_partition_cpu(const Dtype* data_im, const int channels,
     const int height, const int width, const int depth,
	 const int kernel_h, const int kernel_w, const int kernel_d,
     const int pad_h, const int pad_w, const int pad_d,
     const int stride_h, const int stride_w, const int stride_d,
     const int kstride_h, const int kstride_w, const int kstride_d,
	 const size_t start_idx, const size_t end_idx,
     Dtype* data_col){
	 CHECK_GE(start_idx,0)<<": start index must not be negative";
	 CHECK_GE(end_idx,start_idx)<<"end index must be equal or greater than start index";
	  size_t ext_kernel_h = (kernel_h - 1) * kstride_h + 1;
	  size_t ext_kernel_w = (kernel_w - 1) * kstride_w + 1;
	  size_t ext_kernel_d = (kernel_d - 1) * kstride_d + 1;
	  size_t height_out   = (height - ext_kernel_h) / stride_h + 1;
	  size_t width_out    = (width - ext_kernel_w)  / stride_w + 1;
	  size_t depth_out    = (depth - ext_kernel_d)  / stride_d + 1;
	  size_t out_idx_count =  height_out * width_out * depth_out;
	  size_t k_interval    =  end_idx-start_idx;
	  CHECK_LE(start_idx,out_idx_count)<<": start index must be  less than total number of dat points";
	  CHECK_LE(end_idx,out_idx_count+1)<<": end index must be  less than total number of dat points";
	  //LOG(INFO)<<"start idx = " << start_idx << " and end_idx =" <<end_idx; 
	  //LOG(INFO)<<"ext_kernel_h = " << ext_kernel_h << " ext_kernel_w = " <<ext_kernel_w << " ext_kernel_d = " <<ext_kernel_d; 
	  //LOG(INFO)<<"kstride_h = " << kstride_h << " kstride_w = " <<kstride_w << " kstride_d = " <<kstride_d; 
	 // LOG(INFO)<<"k_interval =" <<k_interval;
	  //sleep(1);
	  //CHECK_EQ(end_idx = height_out * width_out * )
      for (size_t index =start_idx; index < end_idx; index++)
	  {	  
			size_t d_out    	= 	index % depth_out;
			size_t w_index  	=  	index / depth_out;
			size_t w_out    	=  	w_index % width_out;	
			size_t h_index  	= 	index / (depth_out * width_out);
			size_t h_out    	= 	h_index % height_out;
			//size_t channel_in 	= 	h_index / height_out;
			//size_t channel_out 	= 	channel_in * kernel_h * kernel_w * kernel_d;
			size_t h_in 		= 	h_out * stride_h - pad_h;
			size_t w_in 		= 	w_out * stride_w - pad_w;
			size_t d_in 		= 	d_out * stride_d - pad_d;
			Dtype* data_col_ptr = 	data_col;
			//size_t d_col_idx =(index-start_idx);
			data_col_ptr += (index-start_idx);//((channel_out * height_col + h_out) * width_col + w_out) * depth_col +d_out;
			const Dtype* data_im_ptr = data_im;
			//LOG(INFO)<<"index =" <<index << " h_out = " <<h_out << " w_out =" <<w_out << " d_out =" <<d_out; 
			//size_t im_idx = ((channel_in * height + h_in) * width + w_in) * depth + d_in;
			//data_im_ptr += ((channel_in * height + h_in) * width + w_in) * depth + d_in;
			size_t im_idx = height*width*depth;
			for (size_t c =0; c<channels;c++){
			     //((c * height + h_in) * width + w_in) * depth + d_in;
				for (size_t i = 0; i < ext_kernel_h; i+=kstride_h) {
				  for (size_t j = 0; j < ext_kernel_w; j+=kstride_w) {
					 for (size_t k = 0; k < ext_kernel_d; k+=kstride_d){
						size_t h = h_in + i;
						size_t w = w_in + j;
						size_t d = d_in + k;
						//size_t idx = ((size_t)i * (size_t)width + (size_t)j) * (size_t)depth+ (size_t)k;
						//size_t idx = (i * width + j) * depth+ k;
						size_t idx = (h * width + w) * depth+ d;
						*data_col_ptr = (h >= 0 && w >= 0 && d >=0 && h < height && w < width && d < depth ) ?
								data_im_ptr[idx] : 0;
						data_col_ptr += k_interval;
					}
				  }
				}
				data_im_ptr += im_idx;
		  }
	} 
  }
	 

	
template void im2col_sk_partition_cpu<float>(const float* data_im, const int channels,
     const int height, const int width, const int depth,
	 const int kernel_h, const int kernel_w, const int kernel_d,
     const int pad_h, const int pad_w, const int pad_d,
     const int stride_h, const int stride_w, const int stride_d,
     const int kstride_h, const int kstride_w, const int kstride_d,
	 const size_t start_idx, const size_t end_idx,
     float* data_col);
	 
	 
	
template void im2col_sk_partition_cpu<double>(const double* data_im, const int channels,
     const int height, const int width, const int depth,
	 const int kernel_h, const int kernel_w, const int kernel_d,
     const int pad_h, const int pad_w, const int pad_d,
     const int stride_h, const int stride_w, const int stride_d,
     const int kstride_h, const int kstride_w, const int kstride_d,
	 const size_t start_idx, const size_t end_idx,
     double* data_col);










template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int depth, const int ksize, const int pad,
    const int stride, Dtype* data_col) {
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col  = (width + 2 * pad - ksize) / stride + 1;
  int depth_col  = (depth + 2 * pad - ksize) / stride + 1;
  int channels_col = channels * ksize * ksize * ksize;
  int t_ksize = ksize * ksize * ksize;
  for (int c = 0; c < channels_col; ++c) {
    int d_offset = c% ksize;
    int w_offset = (c / ksize) % ksize;
    int h_offset = (c / ksize / ksize) % ksize;
    int c_im = c / t_ksize;
    for (int h = 0; h < height_col; ++h) {
		  int h_pad = h * stride - pad + h_offset;
      for (int w = 0; w < width_col; ++w) {	
	      int w_pad = w * stride - pad + w_offset;
	     for(int d = 0 ; d < depth_col ; ++d){
		  int d_pad = d * stride - pad + d_offset;	
		
			if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width && d_pad >= 0 && d_pad < depth)
			  data_col[((c * height_col + h) * width_col + w) * depth_col +d] =
				data_im[((c_im * height + h_pad) * width + w_pad) * depth + d_pad];
			else
			  data_col[((c * height_col + h) * width_col + w) * depth_col +d] = 0;
		  }
       }
    }
  }
}


//For different kernel size	
template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int depth,
	const int ksize_h, 	const int ksize_w, const int ksize_d, 
	const int pad_h, const int pad_w,const int pad_d,
	const int stride_h, const int stride_w, const int stride_d,
	Dtype* data_col){
	int height_col = (height + 2 * pad_h - ksize_h) / stride_h + 1;
  int width_col  = (width + 2 * pad_w - ksize_w) / stride_w + 1;
  int depth_col  = (depth + 2 * pad_d - ksize_d) / stride_d + 1;
  int channels_col = channels * ksize_h * ksize_w * ksize_d;
  int t_ksize = ksize_h * ksize_w * ksize_d;
  for (int c = 0; c < channels_col; ++c) {
    int d_offset = c% ksize_d;
    int w_offset = (c / ksize_d) % ksize_w;
    int h_offset = (c / ksize_d / ksize_w) % ksize_h;
    int c_im = c / t_ksize;
    for (int h = 0; h < height_col; ++h) {
		  int h_pad = h * stride_h - pad_h + h_offset;
      for (int w = 0; w < width_col; ++w) {	
	      int w_pad = w * stride_w - pad_w + w_offset;
	     for(int d = 0 ; d < depth_col ; ++d){
		  int d_pad = d * stride_d - pad_d + d_offset;	
		
			if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width && d_pad >= 0 && d_pad < depth)
			  data_col[((c * height_col + h) * width_col + w) * depth_col +d] =
				data_im[((c_im * height + h_pad) * width + w_pad) * depth + d_pad];
			else
			  data_col[((c * height_col + h) * width_col + w) * depth_col +d] = 0;
		  }
       }
    }
  }
	
	
	
}




// Explicit instantiation
template void im2col_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int depth, const int ksize, const int pad,
    const int stride, float* data_col);
template void im2col_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int depth, const int ksize, const int pad,
    const int stride, double* data_col);
	
	
template 
void im2col_cpu<float>(const float* data_im,  const int channels,
    const int height, const int width, const int depth,
	const int ksize_h, 	const int ksize_w, const int ksize_d, 
	const int pad_h, const int pad_w,const int pad_d,
	const int stride_h, const int stride_w, const int stride_d,
	float* data_col);
template 
void im2col_cpu<double>(const double* data_im,  const int channels,
    const int height, const int width, const int depth,
	const int ksize_h, 	const int ksize_w, const int ksize_d, 
	const int pad_h, const int pad_w,const int pad_d,
	const int stride_h, const int stride_w, const int stride_d,
	double* data_col);	
	


// template <typename Dtype>
// void col2im_cpu(const Dtype* data_col, const int channels,
    // const int height, const int width, const int ksize, const int pad,
    // const int stride, Dtype* data_im) {
  // caffe_set(height * width * channels, Dtype(0), data_im);
  // int height_col = (height + 2 * pad - ksize) / stride + 1;
  // int width_col = (width + 2 * pad - ksize) / stride + 1;
  // int channels_col = channels * ksize * ksize;
  // for (int c = 0; c < channels_col; ++c) {
    // int w_offset = c % ksize;
    // int h_offset = (c / ksize) % ksize;
    // int c_im = c / ksize / ksize;
    // for (int h = 0; h < height_col; ++h) {
      // for (int w = 0; w < width_col; ++w) {
        // int h_pad = h * stride - pad + h_offset;
        // int w_pad = w * stride - pad + w_offset;
        // if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          // data_im[(c_im * height + h_pad) * width + w_pad] +=
              // data_col[(c * height_col + h) * width_col + w];
      // }
    // }
  // }
// }


template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int depth, const int ksize, const int pad,
    const int stride, Dtype* data_im) {
  caffe_set(height * width * depth * channels, Dtype(0), data_im);
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int depth_col  = (depth + 2 * pad - ksize) / stride + 1;
  int channels_col = channels * ksize * ksize * ksize;
  int t_ksize = ksize * ksize * ksize;
  for (int c = 0; c < channels_col; ++c) {
    //int w_offset = c % ksize;
    //int h_offset = (c / ksize) % ksize;
    //int c_im = c / ksize / ksize;
	int d_offset = c% ksize;
    int w_offset = (c / ksize) % ksize;
    int h_offset = (c / ksize / ksize) % ksize;
    int c_im = c / t_ksize;
    for (int h = 0; h < height_col; ++h) {
		 int h_pad = h * stride - pad + h_offset;
      for (int w = 0; w < width_col; ++w) {
	     int w_pad = w * stride - pad + w_offset;
	    for(int d = 0 ; d < depth_col ; ++d){
			 int d_pad = d * stride - pad + d_offset;	
			if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width && d_pad >= 0 && d_pad < depth)
			  data_im[((c_im * height + h_pad) * width + w_pad) *depth + d_pad] +=
				  data_col[((c * height_col + h) * width_col + w) *depth_col +d];
		}
      }
    }
  }
}




template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int depth, 
	const int psize_h, const int psize_w, const int psize_d,
	const int pad_h, const int pad_w, const int pad_d,
	const int stride_h, const int stride_w, const int stride_d,
	Dtype* data_im){
	caffe_set(height * width * depth * channels, Dtype(0), data_im);
  int height_col = (height + 2 * pad_h - psize_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - psize_w) / stride_w + 1;
  int depth_col  = (depth + 2 * pad_d - psize_d) / stride_d + 1;
  int channels_col = channels * psize_h * psize_w * psize_d;
  int t_ksize = psize_h * psize_w * psize_d;
  for (int c = 0; c < channels_col; ++c) {
    //int w_offset = c % ksize;
    //int h_offset = (c / ksize) % ksize;
    //int c_im = c / ksize / ksize;
	int d_offset = c% psize_d;
    int w_offset = (c / psize_d) % psize_w;
    int h_offset = (c / psize_d / psize_w) % psize_h;
    int c_im = c / t_ksize;
    for (int h = 0; h < height_col; ++h) {
		 int h_pad = h * stride_h - pad_h + h_offset;
      for (int w = 0; w < width_col; ++w) {
	     int w_pad = w * stride_w - pad_w + w_offset;
	    for(int d = 0 ; d < depth_col ; ++d){
			 int d_pad = d * stride_d - pad_d + d_offset;	
			if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width && d_pad >= 0 && d_pad < depth)
			  data_im[((c_im * height + h_pad) * width + w_pad) *depth + d_pad] +=
				  data_col[((c * height_col + h) * width_col + w) *depth_col +d];
		}
      }
    }
  }
}




// Explicit instantiation
template void col2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int depth, const int ksize, const int pad,
    const int stride, float* data_im);
	
template void col2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int depth, const int ksize, const int pad,
    const int stride, double* data_im);


	
template void col2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int depth, 
	const int psize_h, const int psize_w, const int psize_d,
	const int pad_h, const int pad_w, const int pad_d,
	const int stride_h, const int stride_w, const int stride_d,
 double* data_im);
 
 template void col2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int depth, 
	const int psize_h, const int psize_w, const int psize_d,
	const int pad_h, const int pad_w, const int pad_d,
	const int stride_h, const int stride_w, const int stride_d,
 float* data_im);
 
 
 
// template void col2im_cpu<double>(const double* data_col, const int channels,
    // const int height, const int width, const int depth, 
	// const int psize_h, const int psize_w, const int psize_d, const int pad,
    // const int stride, double* data_im);

// template void col2im_cpu<float>(const float* data_col, const int channels,
    // const int height, const int width, const int depth, 
	// const int psize_h, const int psize_w, const int psize_d, const int pad,
    // const int stride, float* data_im);

// template void col2im_cpu<double>(const double* data_col, const int channels,
    // const int height, const int width, const int depth, 
	// const int psize_h, const int psize_w, const int psize_d, const int pad,
    // const int stride, double* data_im);

}  // namespace caffe
