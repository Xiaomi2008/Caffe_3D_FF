// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

// template <typename Dtype>
// __global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
    // const int height, const int width, const int ksize, const int pad,
    // const int stride, const int height_col, const int width_col,
    // Dtype* data_col) {
  // CUDA_KERNEL_LOOP(index, n) {
    // int w_out = index % width_col;
    // int h_index = index / width_col;
    // int h_out = h_index % height_col;
    // int channel_in = h_index / height_col;
    // int channel_out = channel_in * ksize * ksize;
    // int h_in = h_out * stride - pad;
    // int w_in = w_out * stride - pad;
    // Dtype* data_col_ptr = data_col;
    // data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
    // const Dtype* data_im_ptr = data_im;
    // data_im_ptr += (channel_in * height + h_in) * width + w_in;
    // for (int i = 0; i < ksize; ++i) {
      // for (int j = 0; j < ksize; ++j) {
        // int h = h_in + i;
        // int w = w_in + j;
        // *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
            // data_im_ptr[i * width + j] : 0;
        // data_col_ptr += height_col * width_col;
      // }
    // }
  // }
// }

template <typename Dtype>
__global__ void im2col_sk_gpu_kernel(const int n, const Dtype* data_im,
    const int height, const int width, const int depth, 
	const int kernel_h, const int kernel_w, const int kernel_d,
    const int ext_kernel_h, const int ext_kernel_w, const int ext_kernel_d,
    const int pad_h, const int pad_w, const int pad_d,
    const int stride_h, const int stride_w, const int stride_d,
    const int kstride_h, const int kstride_w, const int kstride_d,
    const int height_col, const int width_col, const int depth_col,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int d_out    = index % depth_col;
	int w_index  =  index / depth_col;
    int w_out    =  w_index % width_col;	
    int h_index  = index / (depth_col * width_col);
    int h_out    = h_index % height_col;
	//int x= 0;
	int channel_in = h_index / height_col;
	int channel_out = channel_in * kernel_h * kernel_w * kernel_d;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
	int d_in = d_out * stride_d - pad_d;
	 Dtype* data_col_ptr = data_col;
    data_col_ptr += ((channel_out * height_col + h_out) * width_col + w_out) * depth_col +d_out;
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += ((channel_in * height + h_in) * width + w_in) * depth + d_in;
    for (int i = 0; i < ext_kernel_h; i+=kstride_h) {
      for (int j = 0; j < ext_kernel_w; j+=kstride_w) {
	     for (int k = 0; k < ext_kernel_d; k+=kstride_d){
			int h = h_in + i;
			int w = w_in + j;
			int d = d_in + k;
			*data_col_ptr = (h >= 0 && w >= 0 && d >=0 && h < height && w < width && d < depth ) ?
				data_im_ptr[(i * width + j) * depth+ k] : 0;
			data_col_ptr += height_col * width_col * depth_col;
		}
      }
    }
  }
}


template <typename Dtype>
void im2col_sk_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int depth, 
	const int kernel_h, const int kernel_w, const int kernel_d,
    const int pad_h, const int pad_w, const int pad_d, 
	const int stride_h, const int stride_w, const int stride_d,
    const int kstride_h, const int kstride_w, const int kstride_d,
    Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int ext_kernel_h = (kernel_h - 1) * kstride_h + 1;
  int ext_kernel_w = (kernel_w - 1) * kstride_w + 1;
  int ext_kernel_d = (kernel_d - 1) * kstride_d + 1;
  int height_col = (height + 2 * pad_h - ext_kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - ext_kernel_w) / stride_w + 1;
  int depth_col = (depth + 2 * pad_d - ext_kernel_d) / stride_d + 1;
  int num_kernels = channels * height_col * width_col * depth_col;

  //LOG(INFO) << "ext_height = " << ext_kernel_h;
  //LOG(INFO) << "ext_width = " << ext_kernel_w;

  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_sk_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, depth, kernel_h, kernel_w, kernel_d,
      ext_kernel_h, ext_kernel_w, ext_kernel_d, pad_h, pad_w, pad_d, stride_h, stride_w, stride_d,kstride_h, kstride_w, kstride_d,
      height_col, width_col, depth_col,
      data_col);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2col_sk_gpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int depth, 
	const int kernel_h, const int kernel_w, const int kernel_d,
    const int pad_h, const int pad_w, const int pad_d, 
	const int stride_h, const int stride_w, const int stride_d,
    const int kstride_h, const int kstride_w, const int kstride_d,
    float* data_col);
template void im2col_sk_gpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int depth, 
	const int kernel_h, const int kernel_w, const int kernel_d,
    const int pad_h, const int pad_w, const int pad_d, 
	const int stride_h, const int stride_w, const int stride_d,
    const int kstride_h, const int kstride_w, const int kstride_d,
    double* data_col);



	
	
	
	
template <typename Dtype>
__global__ void im2col_sk_gpu_partition_kernel(const size_t n, const Dtype* data_im,
    const int height, const int width, const int depth, 
	const int kernel_h, const int kernel_w, const int kernel_d,
    const int ext_kernel_h, const int ext_kernel_w, const int ext_kernel_d,
    const int pad_h, const int pad_w, const int pad_d,
    const int stride_h, const int stride_w, const int stride_d,
    const int kstride_h, const int kstride_w, const int kstride_d,
    const int height_col, const int width_col, const int depth_col,size_t start_idx, size_t end_idx, const int in_channels,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    size_t in_full_size = (size_t)height_col*(size_t)width_col*(size_t)depth_col;
    size_t ch_size  = end_idx - start_idx;
	size_t out_ch   = index / ch_size;
	size_t n_out    = (size_t)index%ch_size;
	size_t out_index= out_ch*in_full_size+start_idx+n_out;
	
    size_t cur_idx  = out_index ;//+start_idx;//*(size_t)in_channels;
    size_t d_out    = cur_idx % depth_col;
	size_t w_index  =  cur_idx / depth_col;
    size_t w_out    =  w_index % width_col;	
    size_t h_index  = cur_idx / (depth_col * width_col);
    size_t h_out    = h_index % height_col;
	//int x= 0;
	size_t channel_in = h_index / height_col;
	//size_t channel_out = (size_t)channel_in * (size_t)kernel_h * (size_t)kernel_w * (size_t)kernel_d;
	size_t channel_out = (size_t)out_ch * (size_t)kernel_h * (size_t)kernel_w * (size_t)kernel_d;
	//size_t channel_out = (size_t)index/ch_size;//(size_t)kernel_h * (size_t)kernel_w * (size_t)kernel_d;
	
    //size_t h_in = h_out * stride_h - pad_h;
    //size_t w_in = w_out * stride_w - pad_w;
	//size_t d_in = d_out * stride_d - pad_d;
	
	int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
	int d_in = d_out * stride_d - pad_d;
	Dtype* data_col_ptr = data_col;
   // data_col_ptr += ((channel_out * (size_t)height_col + h_out) * (size_t)width_col + w_out) * (size_t)depth_col +d_out;
    size_t d_offset =(channel_out*ch_size+n_out);
    data_col_ptr+=d_offset;
    const Dtype* data_im_ptr = data_im;
   data_im_ptr += ((channel_in * (size_t)height + h_in) * (size_t)width + w_in) * (size_t)depth + d_in;
   //data_im_ptr +=channel_in *((size_t)height * (size_t)width * (size_t)depth);
    for (size_t i = 0; i < ext_kernel_h; i+=kstride_h) {
      for (size_t j = 0; j < ext_kernel_w; j+=kstride_w) {
	     for (size_t k = 0; k < ext_kernel_d; k+=kstride_d){
			//size_t h = h_in + i;
			//size_t w = w_in + j;
			//size_t d = d_in + k;
			int h = h_in + i;
			int w = w_in + j;
			int d = d_in + k;
			size_t idx = (i * (size_t)width + j) * (size_t)depth+ k;
			//*data_col_ptr = (h >= 0 && w >= 0 && d >=0 && h < height && w < width && d < depth ) ?
			//	data_im_ptr[(i * (size_t)width + j) * (size_t)depth+ k] : 0;
			
			
			*data_col_ptr = (h >= 0 && w >= 0 && d >=0 && h < height && w < width && d < depth ) ?
				data_im_ptr[idx] : 0;
			//Dtype x=(h >= 0 && w >= 0 && d >=0 && h < height && w < width && d < depth ) ?
			//	 data_im_ptr[idx]:0;
			//*data_col_ptr=2.2;
			data_col_ptr +=ch_size;/// height_col * width_col * depth_col;
		}
      }
    }
  }
}	

	
template <typename Dtype>
void im2col_sk_partition_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int depth, 
	const int kernel_h, const int kernel_w, const int kernel_d,
    const int pad_h, const int pad_w, const int pad_d, 
	const int stride_h, const int stride_w, const int stride_d,
    const int kstride_h, const int kstride_w, const int kstride_d,
	const size_t start_idx, const size_t end_idx,
    Dtype* data_col){
	  size_t ext_kernel_h = (kernel_h - 1) * kstride_h + 1;
	  size_t ext_kernel_w = (kernel_w - 1) * kstride_w + 1;
	  size_t ext_kernel_d = (kernel_d - 1) * kstride_d + 1;
	  size_t height_col = (height + 2 * pad_h - ext_kernel_h) / stride_h + 1;
	  size_t width_col = (width + 2 * pad_w - ext_kernel_w) / stride_w + 1;
	  size_t depth_col = (depth + 2 * pad_d - ext_kernel_d) / stride_d + 1;
	  size_t num_kernels = channels*(end_idx-start_idx); //* height_col * width_col * depth_col;

	  // NOLINT_NEXT_LINE(whitespace/operators)
	  im2col_sk_gpu_partition_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
								 CAFFE_CUDA_NUM_THREADS>>>(
		  num_kernels, data_im, height, width, depth, kernel_h, kernel_w, kernel_d,
		  ext_kernel_h, ext_kernel_w, ext_kernel_d, pad_h, pad_w, pad_d, stride_h, stride_w, stride_d,kstride_h, kstride_w, kstride_d,
		  height_col, width_col, depth_col,start_idx,end_idx,channels,
		  data_col);
	  CUDA_POST_KERNEL_CHECK;
	
}	

template void im2col_sk_partition_gpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int depth, 
	const int kernel_h, const int kernel_w, const int kernel_d,
    const int pad_h, const int pad_w, const int pad_d, 
	const int stride_h, const int stride_w, const int stride_d,
    const int kstride_h, const int kstride_w, const int kstride_d,
	const size_t start_idx, const size_t end_idx,
    double* data_col);
	

template void im2col_sk_partition_gpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int depth, 
	const int kernel_h, const int kernel_w, const int kernel_d,
    const int pad_h, const int pad_w, const int pad_d, 
	const int stride_h, const int stride_w, const int stride_d,
    const int kstride_h, const int kstride_w, const int kstride_d,
	const size_t start_idx, const size_t end_idx,
    float* data_col);
	
	
	


template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
    const int height, const int width, const int depth, const int ksize, const int pad,
    const int stride, const int height_col, const int width_col, const int depth_col,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int d_out    = index % depth_col;
	int w_index  =  index / depth_col;
    int w_out    =  w_index % width_col;	
    int h_index  = index / (depth_col * width_col);
    int h_out    = h_index % height_col;
    int channel_in = h_index / height_col;
    int channel_out = channel_in * ksize * ksize * ksize;
    int h_in = h_out * stride - pad;
    int w_in = w_out * stride - pad;
	int d_in = d_out * stride - pad;
    Dtype* data_col_ptr = data_col;
    data_col_ptr += ((channel_out * height_col + h_out) * width_col + w_out) * depth_col +d_out;
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += ((channel_in * height + h_in) * width + w_in) * depth + d_in;
    for (int i = 0; i < ksize; ++i) {
      for (int j = 0; j < ksize; ++j) {
	     for (int k = 0; k < ksize; ++k){
			int h = h_in + i;
			int w = w_in + j;
			int d = d_in + k;
			*data_col_ptr = (h >= 0 && w >= 0 && d >=0 && h < height && w < width && d < depth ) ?
				data_im_ptr[(i * width + j) * depth+ k] : 0;
			data_col_ptr += height_col * width_col * depth_col;
		}
      }
    }
  }
}


// template <typename Dtype>
// void im2col_gpu(const Dtype* data_im, const int channels,
    // const int height, const int width, const int ksize, const int pad,
    // const int stride, Dtype* data_col) {
  // // We are going to launch channels * height_col * width_col kernels, each
  // // kernel responsible for copying a single-channel grid.
  // int height_col = (height + 2 * pad - ksize) / stride + 1;
  // int width_col = (width + 2 * pad - ksize) / stride + 1;
  // int num_kernels = channels * height_col * width_col;
  // // NOLINT_NEXT_LINE(whitespace/operators)
  // im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             // CAFFE_CUDA_NUM_THREADS>>>(
      // num_kernels, data_im, height, width, ksize, pad, stride, height_col,
      // width_col, data_col);
  // CUDA_POST_KERNEL_CHECK;
// }


template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int depth, const int ksize, const int pad,
    const int stride, Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int depth_col = (depth + 2 * pad - ksize) / stride + 1;
  int num_kernels = channels * height_col * width_col * depth_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, depth, ksize, pad, stride, height_col,
      width_col, depth_col, data_col);
  CUDA_POST_KERNEL_CHECK;
}



// Explicit instantiation
// template void im2col_gpu<float>(const float* data_im, const int channels,
    // const int height, const int width, const int ksize, const int pad,
    // const int stride, float* data_col);
// template void im2col_gpu<double>(const double* data_im, const int channels,
    // const int height, const int width, const int ksize, const int pad,
    // const int stride, double* data_col);
template void im2col_gpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int depth,
	const int ksize_h, const int ksize_w, const int ksize_d, 
	const int pad_h, const int pad_w, const int pad_d,
    const int stride_h, const int stride_w, const int stride_d, 
	double* data_col);
	
template void im2col_gpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int depth,
	const int ksize_h, const int ksize_w, const int ksize_d, 
	const int pad_h, const int pad_w, const int pad_d,
    const int stride_h, const int stride_w, const int stride_d, 
	float* data_col);
	
template void im2col_gpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int depth, const int ksize, const int pad,
    const int stride, float* data_col);
	
template void im2col_gpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int depth, const int ksize, const int pad,
    const int stride, double* data_col);
	
	

// for diff kernel size
//template void im2col_gpu<double>(const double* data_im, const int channels,
//    const int height, const int width, const int depth, const int ksize, const int pad,
//    const int stride, double* data_col);
	
template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
    const int height, const int width, const int depth, 
	const int ksize_h, const int ksize_w, const int ksize_d,
	const int pad_h, const int pad_w, const int pad_d,
	const int stride_h, const int stride_w, const int stride_d, 
	const int height_col, const int width_col,	const int depth_col,  
	Dtype* data_col) 
 {
  CUDA_KERNEL_LOOP(index, n) {
    int d_out    = index % depth_col;
	int w_index  =  index / depth_col;
    int w_out    =  w_index % width_col;	
    int h_index  = index / (depth_col * width_col);
    int h_out    = h_index % height_col;
    int channel_in = h_index / height_col;
    int channel_out = channel_in * ksize_h * ksize_w * ksize_d;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
	int d_in = d_out * stride_d - pad_d;
    Dtype* data_col_ptr = data_col;
    data_col_ptr += ((channel_out * height_col + h_out) * width_col + w_out) * depth_col +d_out;
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += ((channel_in * height + h_in) * width + w_in) * depth + d_in;
    for (int i = 0; i < ksize_h; ++i) {
      for (int j = 0; j < ksize_w; ++j) {
	     for (int k = 0; k < ksize_d; ++k){
			int h = h_in + i;
			int w = w_in + j;
			int d = d_in + k;
			*data_col_ptr = (h >= 0 && w >= 0 && d >=0 && h < height && w < width && d < depth ) ?
				data_im_ptr[(i * width + j) * depth+ k] : 0;
			//*data_col_ptr = (h >= 0 && w >= 0 && d >=0 && h < height && w < width && d < depth ) ?
			//	data_im_ptr[(i * depth + j) * width+ k] : 0;
				//data_im_ptr[i * width + j] : 0;
			data_col_ptr += height_col * width_col * depth_col;
		}
      }
    }
  }
}

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int depth, 
	const int ksize_h, const int ksize_w, const int ksize_d, 
	const int pad_h,  const int pad_w, const int pad_d,
	const int stride_h, const int stride_w, const int stride_d,
	Dtype* data_col){
	
	
  int height_col = (height + 2 * pad_h - ksize_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - ksize_w) / stride_w + 1;
  int depth_col = (depth + 2 * pad_d - ksize_d) / stride_d + 1;
  int num_kernels = channels * height_col * width_col * depth_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, depth, 
	  ksize_h,ksize_w, ksize_d,
	  pad_h, pad_w, pad_d, 
	  stride_h, stride_w, stride_d, 
	  height_col,  width_col, depth_col, 
	  data_col);
  CUDA_POST_KERNEL_CHECK;
	
}



// template 
// void im2col_gpu<float>(const float* data_im, const int channels,
    // const int height, const int width, const int depth,
	// const int ksize_h, 	const int ksize_w, const int ksize_d, 
	// const int pad, const int stride, float* data_col);
// template 
// void im2col_gpu<double>(const double* data_im, const int channels,
    // const int height, const int width, const int depth,
	// const int ksize_h, 	const int ksize_w, const int ksize_d, 
	// const int pad, const int stride, double* data_col);	
	
	
	

// template <typename Dtype>
// __global__ void col2im_gpu_kernel(const int n, const Dtype* data_col,
    // const int height, const int width, const int channels, const int ksize,
    // const int pad, const int stride, const int height_col, const int width_col,
    // Dtype* data_im) {
  // CUDA_KERNEL_LOOP(index, n) {
    // Dtype val = 0;
    // int w = index % width + pad;
    // int h = (index / width) % height + pad;
    // int c = index / (width * height);
    // // compute the start and end of the output
    // int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
    // int w_col_end = min(w / stride + 1, width_col);
    // int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
    // int h_col_end = min(h / stride + 1, height_col);
    // /*
    // for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      // for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        // // the col location: [c * width * height + h_out, w_out]
        // int c_col = c * ksize * ksize + (h - h_col * stride) * ksize + (w - w_col * stride);
        // val += data_col[(c_col * height_col + h_col) * width_col + w_col];
      // }
    // }
    // */
    // // equivalent implementation
    // int offset = (c * ksize * ksize + h * ksize + w) * height_col * width_col;
    // int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
    // int coeff_w_col = (1 - stride * height_col * width_col);
    // for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      // for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        // val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      // }
    // }
    // data_im[index] = val;
  // }
// }

template <typename Dtype>
__global__ void col2im_gpu_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int depth, const int channels, const int ksize,
    const int pad, const int stride, const int height_col, const int width_col, const int depth_col,
    Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
	int d = index % depth + pad;
    int w = (index /depth) % width + pad;
    int h = (index / (depth * width)) % height + pad;
    int c = index / (depth * width * height );
    // compute the start and end of the output
	int d_col_start = (d < ksize) ? 0 : (d - ksize) / stride + 1;
    int d_col_end = min(d / stride + 1, depth_col);
    int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
    int w_col_end = min(w / stride + 1, width_col);
    int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
    int h_col_end = min(h / stride + 1, height_col);
    int k3=ksize * ksize *ksize;
	int k2=ksize * ksize;
     for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
	    for (int d_col = d_col_start; d_col < d_col_end; ++d_col) {
			// the col location: [c * width * height + h_out, w_out]
			//int c_col = c * ksize * ksize + (h - h_col * stride) * ksize + (w - w_col * stride);
			//int c_col = c *  + (h - h_col * stride) * ksize *ksize + (w - w_col * stride)*ksize + (d - d_col * stride);
			int c_col = c * k3 + (h - h_col * stride) * k2 + (w - w_col * stride)*ksize + (d - d_col * stride);
			
			val += data_col[((c_col * height_col + h_col) * width_col + w_col) * depth_col +d_col];
			//int a = c* +1;
		}
      }
    }
    
    // equivalent implementation
    //int offset = (c * ksize * ksize + h * ksize + w) * height_col * width_col;
	/*
	//int offset = (c * ksize * ksize * ksize + h * ksize *ksize + w * ksize + d) * height_col * width_col * depth_col;
    //int coeff_h_col = (1 - stride * ksize * ksize * height_col) * width_col * ksize * depth_col; // most likely wrong position
   // int coeff_w_col = (1 - stride * height_col * width_col);
   
    //int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
    //int coeff_w_col = (1 - stride_w * height_col * width_col);
   
	int coeff_w_col = (1 - stride * ksize * width_col) * depth_col;
	int coeff_d_col = (1 - stride * width_col * depth_col);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
	    for (int d_col = d_col_start; d_col < d_col_end; ++d_col){
			val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col + d_col * coeff_d_col];
			}
      }
    }*/
    data_im[index] = val;
  }
}



// template <typename Dtype>
// void col2im_gpu(const Dtype* data_col, const int channels,
    // const int height, const int width, const int ksize, const int pad,
    // const int stride, Dtype* data_im) {
  // int height_col = (height + 2 * pad - ksize) / stride + 1;
  // int width_col = (width + 2 * pad - ksize) / stride + 1;
  // int num_kernels = channels * height * width;
  // // To avoid involving atomic operations, we will launch one kernel per
  // // bottom dimension, and then in the kernel add up the top dimensions.
  // // NOLINT_NEXT_LINE(whitespace/operators)
  // col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             // CAFFE_CUDA_NUM_THREADS>>>(
      // num_kernels, data_col, height, width, channels, ksize, pad, stride,
      // height_col, width_col, data_im);
  // CUDA_POST_KERNEL_CHECK;
// }


template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int depth, const int ksize, const int pad,
    const int stride, Dtype* data_im) {
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int depth_col = (depth + 2 * pad - ksize) / stride + 1;
  int num_kernels = channels * height * width * depth;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, depth, channels, ksize, pad, stride,
      height_col, width_col, depth_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}












template <typename Dtype>
__global__ void col2im_gpu_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int depth, const int channels, 
	const int ksize_h,const int ksize_w,const int ksize_d,
    const int pad_h, const int pad_w, const int pad_d, 
	const int stride_h, const int stride_w, const int stride_d,
	const int height_col, const int width_col, const int depth_col,
    Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
	int d = index % depth + pad_d;
    int w = (index /depth) % width + pad_w;
    int h = (index / (depth * width)) % height + pad_h;
    int c = index / (depth * width * height );
    // compute the start and end of the output
	int d_col_start = (d < ksize_d) ? 0 : (d - ksize_d) / stride_d + 1;
    int d_col_end = min(d / stride_d + 1, depth_col);
    int w_col_start = (w < ksize_w) ? 0 : (w - ksize_w) / stride_w + 1;
    int w_col_end = min(w / stride_w + 1, width_col);
    int h_col_start = (h < ksize_h) ? 0 : (h - ksize_h) / stride_h + 1;
    int h_col_end = min(h / stride_h + 1, height_col);
    int k3=ksize_h * ksize_w *ksize_d;
	int k2=ksize_w * ksize_d;
     for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
	    for (int d_col = d_col_start; d_col < d_col_end; ++d_col) {
			// the col location: [c * width * height + h_out, w_out]
			//int c_col = c * ksize * ksize + (h - h_col * stride) * ksize + (w - w_col * stride);
			//int c_col = c *  + (h - h_col * stride) * ksize *ksize + (w - w_col * stride)*ksize + (d - d_col * stride);
			int c_col = c * k3 + (h - h_col * stride_h) * k2 + (w - w_col * stride_w)*ksize_w + (d - d_col * stride_d);
			
			val += data_col[((c_col * height_col + h_col) * width_col + w_col) * depth_col +d_col];
			//int a = c* +1;
		}
      }
    }
    data_im[index] = val;
  }
}








template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int depth, 
	const int psize_h, const int psize_w, const int psize_d, 
	const int pad_h, const int pad_w, const int pad_d,
	const int stride_h, const int stride_w, const int stride_d, 
	Dtype* data_im){
	int height_col = (height + 2 * pad_h - psize_h) / stride_h + 1;
	int width_col = (width + 2 * pad_w - psize_w) / stride_w + 1;
	int depth_col = (depth + 2 * pad_d - psize_d) / stride_d + 1;
	int num_kernels = channels * height * width * depth;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, depth, channels, 
	  psize_h, psize_w, psize_d, 
	  pad_h, pad_w,pad_d, stride_h, stride_w, stride_d,
      height_col, width_col, depth_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}



// // Explicit instantiation
// template void col2im_gpu<float>(const float* data_col, const int channels,
    // const int height, const int width, const int psize, const int pad,
    // const int stride, float* data_im);
// template void col2im_gpu<double>(const double* data_col, const int channels,
    // const int height, const int width, const int psize, const int pad,
    // const int stride, double* data_im);
	
// Explicit instantiation
template void col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int depth, const int psize, const int pad,
    const int stride, float* data_im);
template void col2im_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int depth, const int psize, const int pad,
    const int stride, double* data_im);

template void col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int depth, 
	const int psize_h, const int psize_w, const int psize_d, 
	const int pad_h, const int pad_w, const int pad_d,
	const int stride_h, const int stride_w, const int stride_d, float* data_im);

template void col2im_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int depth, 
	const int psize_h, const int psize_w, const int psize_d, 
	const int pad_h, const int pad_w, const int pad_d,
	const int stride_h, const int stride_w, const int stride_d, double* data_im);


	



template <typename Dtype>
__global__ void col2im_sk_gpu_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int depth, const int channels, 
	const int ksize_h,const int ksize_w,const int ksize_d,
	const int ext_kernel_h,const int ext_kernel_w,const int ext_kernel_d,
    const int pad_h, const int pad_w, const int pad_d, 
	const int stride_h, const int stride_w, const int stride_d,
	const int kstride_h, const int kstride_w, const int kstride_d,
	const int height_col, const int width_col, const int depth_col,
    Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
	int d = index % depth + pad_d;
    int w = (index /depth) % width + pad_w;
    int h = (index / (depth * width)) % height + pad_h;
    int c = index / (depth * width * height );
    // compute the start and end of the output
	
	size_t ext_d_col_start = (d < ext_kernel_d) ? 0 : (d - ext_kernel_d) / stride_d + 1;
    size_t ext_d_col_end = min(d / stride_d + 1, depth_col);
    size_t ext_w_col_start = (w < ext_kernel_w) ? 0 : (w - ext_kernel_w) / stride_w + 1;
    size_t ext_w_col_end = min(w / stride_w + 1, width_col);
    size_t ext_h_col_start = (h < ext_kernel_h) ? 0 : (h - ext_kernel_h) / stride_h + 1;
    size_t ext_h_col_end = min(h / stride_h + 1, height_col);
    // size_t k3=ksize_h * ksize_w *ksize_d;
	// size_t k2=ksize_w * ksize_d;
	size_t d_col_start = (d < ksize_d) ? 0 : (d - ksize_d) / stride_d + 1;
    size_t d_col_end = min(d / stride_d + 1, depth_col);
    size_t w_col_start = (w < ksize_w) ? 0 : (w - ksize_w) / stride_w + 1;
    size_t w_col_end = min(w / stride_w + 1, width_col);
    size_t h_col_start = (h < ksize_h) ? 0 : (h - ksize_h) / stride_h + 1;
    size_t h_col_end = min(h / stride_h + 1, height_col);
    size_t k3=ksize_h * ksize_w *ksize_d;
	size_t k2=ksize_w * ksize_d;
	size_t ext_h_col =  ext_h_col_start;
     for (size_t h_col = h_col_start; h_col < h_col_end; ++h_col) {
	       size_t ext_w_col =  ext_w_col_start;
     for (size_t w_col = w_col_start; w_col < w_col_end; ++w_col) {
	       size_t ext_d_col =  ext_d_col_start;
	   for (size_t d_col = d_col_start; d_col < d_col_end; ++d_col) {
	 // for (size_t h_col = h_col_start; h_col < h_col_end; h_col+=kstride_h) {
      // for (size_t w_col = w_col_start; w_col < w_col_end; w_col+=kstride_w) {
	    // for (size_t d_col = d_col_start; d_col < d_col_end; d_col+=kstride_d) {
			// the col location: [c * width * height + h_out, w_out]
			//int c_col = c * ksize * ksize + (h - h_col * stride) * ksize + (w - w_col * stride);
			//int c_col = c *  + (h - h_col * stride) * ksize *ksize + (w - w_col * stride)*ksize + (d - d_col * stride);
			//size_t c_col = c * k3 + (h - h_col * stride_h) * k2 + (w - w_col * stride_w)*ksize_w + (d - //d_col * stride_d);
			
			size_t c_col = c * k3 + (h - h_col* stride_h) * k2 + (w - w_col* stride_w)*ksize_w + (d - d_col * stride_d);
			//CHECK_GE(c_col,0);
			//if(c_col<0){LOG(INFO)<<"col index is negative";}
			size_t d_idx =((c_col * height_col + ext_h_col) * width_col + ext_w_col) * depth_col +ext_d_col;
			//if(d_idx<0){LOG(INFO)<<"index is negative";}
			val += data_col[d_idx];
			ext_d_col+=kstride_d;
			
			
			// other implementation
				// int offset =
					// (c * patch_h * patch_w + h * patch_w + w) * height_col * width_col;
				// int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
				// int coeff_w_col = (1 - stride_w * height_col * width_col);
				// for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
					// for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
						// val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
					// }		
				// }
			
		}
		ext_w_col+=kstride_w;
      }
	   ext_h_col+=kstride_h;
    }
    data_im[index] = val;
  }
  
  
	// int d_out    = index % depth_col;
	// int w_index  =  index / depth_col;
    // int w_out    =  w_index % width_col;	
    // int h_index  = index / (depth_col * width_col);
    // int h_out    = h_index % height_col;
	// //int x= 0;
	// int channel_in = h_index / height_col;
	// int channel_out = channel_in * kernel_h * kernel_w * kernel_d;
    // int h_in = h_out * stride_h - pad_h;
    // int w_in = w_out * stride_w - pad_w;
	// int d_in = d_out * stride_d - pad_d;
	 // Dtype* data_col_ptr = data_col;
    // data_col_ptr += ((channel_out * height_col + h_out) * width_col + w_out) * depth_col +d_out;
    // const Dtype* data_im_ptr = data_im;
    // data_im_ptr += ((channel_in * height + h_in) * width + w_in) * depth + d_in;
    // for (int i = 0; i < ext_kernel_h; i+=kstride_h) {
      // for (int j = 0; j < ext_kernel_w; j+=kstride_w) {
	     // for (int k = 0; k < ext_kernel_d; k+=kstride_d){
			// int h = h_in + i;
			// int w = w_in + j;
			// int d = d_in + k;
			// *data_col_ptr = (h >= 0 && w >= 0 && d >=0 && h < height && w < width && d < depth ) ?
				// data_im_ptr[(i * width + j) * depth+ k] : 0;
			// data_col_ptr += height_col * width_col * depth_col;
		// }
      // }
    // }

  
  
  
}


template <typename Dtype>
void col2im_sk_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int depth, 
	const int kernel_h, const int kernel_w, const int kernel_d, 
	const int pad_h, const int pad_w, const int pad_d,
	const int stride_h, const int stride_w, const int stride_d,
    const int kstride_h, const int kstride_w, const int kstride_d,	
	Dtype* data_im){
			// We are going to launch channels * height_col * width_col kernels, each
	  // kernel responsible for copying a single-channel grid.
	  int ext_kernel_h = (kernel_h - 1) * kstride_h + 1;
	  int ext_kernel_w = (kernel_w - 1) * kstride_w + 1;
	  int ext_kernel_d = (kernel_d - 1) * kstride_d + 1;
	  int height_col = (height + 2 * pad_h - ext_kernel_h) / stride_h + 1;
	  int width_col = (width + 2 * pad_w - ext_kernel_w) / stride_w + 1;
	  int depth_col = (depth + 2 * pad_d - ext_kernel_d) / stride_d + 1;
	  //int num_kernels = channels * height_col * width_col * depth_col;
	  int num_kernels = channels * height * width * depth;
	  
	
	  // To avoid involving atomic operations, we will launch one kernel per
	  // bottom dimension, and then in the kernel add up the top dimensions.
	  // NOLINT_NEXT_LINE(whitespace/operators)
	  col2im_sk_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
								 CAFFE_CUDA_NUM_THREADS>>>(
		  num_kernels, data_col, height, width, depth, channels, 
		  kernel_h, kernel_w, kernel_d, ext_kernel_h,ext_kernel_w, ext_kernel_d,
		  pad_h, pad_w, pad_d, stride_h, stride_w, stride_d,
		  kstride_h, kstride_w, kstride_d,
		  height_col, width_col, depth_col, data_im);
	  CUDA_POST_KERNEL_CHECK;
	
	}
	

	
	
	
template	
void col2im_sk_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int depth, 
	const int psize_h, const int psize_w, const int psize_d, 
	const int pad_h, const int pad_w, const int pad_d,
	const int stride_h, const int stride_w, const int stride_d,
    const int kstride_h, const int kstride_w, const int kstride_d,	
	float* data_im);

template	
void col2im_sk_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int depth, 
	const int psize_h, const int psize_w, const int psize_d, 
	const int pad_h, const int pad_w, const int pad_d,
	const int stride_h, const int stride_w, const int stride_d,
    const int kstride_h, const int kstride_w, const int kstride_d,	
	double* data_im);
	
	
	
	
// template <typename Dtype>
// void im2col_sk_partition_gpu(const Dtype* data_im, const int channels,
     // const int height, const int width, const int depth,
	 // const int kernel_h, const int kernel_w, const int kernel_d,
     // const int pad_h, const int pad_w, const int pad_d,
     // const int stride_h, const int stride_w, const int stride_d,
     // const int kstride_h, const int kstride_w, const int kstride_d,
	 // const int start_idx, const int end_idx,
     // Dtype* data_col){
	 // CHECK_GE(start_idx,0)<<": start index must not be negative";
	 // CHECK_GE(end_idx,start_idx)<<"end index must be equal or greater than start index";
	  // int ext_kernel_h = (kernel_h - 1) * kstride_h + 1;
	  // int ext_kernel_w = (kernel_w - 1) * kstride_w + 1;
	  // int ext_kernel_d = (kernel_d - 1) * kstride_d + 1;
	  // int height_out   = (height - ext_kernel_h) / stride_h + 1;
	  // int width_out    = (width - ext_kernel_w)  / stride_w + 1;
	  // int depth_out    = (depth - ext_kernel_d)  / stride_d + 1;
	  // int out_idx_count =  height_out * width_out * depth_out;
	  // int k_interval    = end_idx-start_idx;
	  // CHECK_LE(start_idx,out_idx_count)<<": start index must be  less than total number of dat points";
	  // CHECK_LE(end_idx,out_idx_count)<<": end index must be  less than total number of dat points";
	  
      // for (int index =start_idx; index < end_idx; index++)
	  // {	  
			// int d_out    = index % depth_out;
			// int w_index  =  index / depth_out;
			// int w_out    =  w_index % width_out;	
			// int h_index  = index / (depth_out * width_out);
			// int h_out    = h_index % height_out;
			// int channel_in = h_index / height_out;
			// int channel_out = channel_in * kernel_h * kernel_w * kernel_d;
			// int h_in = h_out * stride_h - pad_h;
			// int w_in = w_out * stride_w - pad_w;
			// int d_in = d_out * stride_d - pad_d;
			 // Dtype* data_col_ptr = data_col;
			// //size_t d_col_idx =(index-start_idx);
			// data_col_ptr += (index-start_idx);
			// const Dtype* data_im_ptr = data_im;
			// size_t im_idx = (((size_t)channel_in * (size_t)height + (size_t)h_in) * (size_t)width + (size_t)w_in) * (size_t)depth + (size_t)d_in;
			// //data_im_ptr += ((channel_in * height + h_in) * width + w_in) * depth + d_in;
			// data_im_ptr += im_idx;
			// for (int i = 0; i < ext_kernel_h; i+=kstride_h) {
			  // for (int j = 0; j < ext_kernel_w; j+=kstride_w) {
				 // for (int k = 0; k < ext_kernel_d; k+=kstride_d){
					// int h = h_in + i;
					// int w = w_in + j;
					// int d = d_in + k;
					// size_t idx = ((size_t)i * (size_t)width + (size_t)j) * (size_t)depth+ (size_t)k;
					// *data_col_ptr = (h >= 0 && w >= 0 && d >=0 && h < height && w < width && d < depth ) ?
				    		// data_im_ptr[idx] : 0;			
					// data_col_ptr += k_interval;
					
				// }
			  // }
			// }
		// }
		  
  
  // }
	 

	
// template void im2col_sk_partition_gpu<float>(const float* data_im, const int channels,
     // const int height, const int width, const int depth,
	 // const int kernel_h, const int kernel_w, const int kernel_d,
     // const int pad_h, const int pad_w, const int pad_d,
     // const int stride_h, const int stride_w, const int stride_d,
     // const int kstride_h, const int kstride_w, const int kstride_d,
	 // const int part_idx, const int part_num,
     // float* data_col);
	 
	 
	
// template void im2col_sk_partition_gpu<double>(const double* data_im, const int channels,
     // const int height, const int width, const int depth,
	 // const int kernel_h, const int kernel_w, const int kernel_d,
     // const int pad_h, const int pad_w, const int pad_d,
     // const int stride_h, const int stride_w, const int stride_d,
     // const int kstride_h, const int kstride_w, const int kstride_d,
	 // const int part_idx, const int part_num,
     // double* data_col);


}  // namespace caffe
