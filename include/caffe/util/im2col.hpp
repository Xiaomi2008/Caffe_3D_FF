// Copyright 2014 BVLC and contributors.

#ifndef _CAFFE_UTIL_IM2COL_HPP_
#define _CAFFE_UTIL_IM2COL_HPP_

namespace caffe {

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int depth,const int ksize, const int pad,
    const int stride, Dtype* data_col);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int depth, const int psize, const int pad,
    const int stride, Dtype* data_im);

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int depth, const int ksize, const int pad,
    const int stride, Dtype* data_col);

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int depth, const int psize, const int pad,
    const int stride, Dtype* data_im);
	

	

//For different kernel size	
template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int depth,
	const int ksize_h, 	const int ksize_w, const int ksize_d, 
	const int pad_h, const int pad_w,const int pad_d,
	const int stride_h, const int stride_w, const int stride_d,
	Dtype* data_col);
	
template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int depth, 
	const int psize_h, const int psize_w, const int psize_d,
	const int pad_h, const int pad_w, const int pad_d,
	const int stride_h, const int stride_w, const int stride_d,
	Dtype* data_im);
	
template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int depth, 
	const int ksize_h, const int ksize_w, const int ksize_d, 
	const int pad_h,  const int pad_w, const int pad_d,
	const int stride_h, const int stride_w, const int stride_d,
	Dtype* data_col);
	
template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int depth, 
	const int psize_h, const int psize_w, const int psize_d, 
	const int pad_h, const int pad_w, const int pad_d,
	const int stride_h, const int stride_w, const int stride_d, 
	Dtype* data_im);
	
	
	
template <typename Dtype>
void col2im_sk_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int depth, 
	const int psize_h, const int psize_w, const int psize_d, 
	const int pad_h, const int pad_w, const int pad_d,
	const int stride_h, const int stride_w, const int stride_d,
    const int kstride_h, const int kstride_w, const int kstride_d,	
	Dtype* data_im);
	
	

	
// template <typename Dtype>
// void im2col_cpu(const Dtype* data_im, const int channels,
    // const int height, const int width, const int depth,const int ksize_h, const int ksize_w, 
	// const int ksize_d, const int pad,
    // const int stride, Dtype* data_col);


// template <typename Dtype>
// void col2im_cpu(const Dtype* data_col, const int channels,
    // const int height, const int width, const int depth, 
	// const int psize_h, const int psize_w, const int psize_d,
	// const int pad,const int stride, Dtype* data_im);

// template <typename Dtype>
// void im2col_gpu(const Dtype* data_im, const int channels,
    // const int height, const int width, const int depth, 
	// const int ksize_h, const int ksize_w, const int ksize_d, 
	// const int pad, const int stride, Dtype* data_col);

// template <typename Dtype>
// void col2im_gpu(const Dtype* data_col, const int channels,
    // const int height, const int width, const int depth, 
	// const int psize_h, const int psize_w, const int psize_d, 
	// const int pad, const int stride, Dtype* data_im);

template <typename Dtype>
void im2col_sk_cpu(const Dtype* data_im, const int channels,
     const int height, const int width, const int depth,
	 const int kernel_h, const int kernel_w, const int kernel_d,
     const int pad_h, const int pad_w, const int pad_d,
     const int stride_h, const int stride_w, const int stride_d,
     const int kstride_h, const int kstride_w, const int kstride_d,
     Dtype* data_col);
	 
template <typename Dtype>	 
void im2col_sk_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int depth, 
	const int kernel_h, const int kernel_w, const int kernel_d,
    const int pad_h, const int pad_w, const int pad_d, 
	const int stride_h, const int stride_w, const int stride_d,
    const int kstride_h, const int kstride_w, const int kstride_d,
    Dtype* data_col);
	
	
template <typename Dtype>
void im2col_sk_partition_cpu(const Dtype* data_im, const int channels,
     const int height, const int width, const int depth,
	 const int kernel_h, const int kernel_w, const int kernel_d,
     const int pad_h, const int pad_w, const int pad_d,
     const int stride_h, const int stride_w, const int stride_d,
     const int kstride_h, const int kstride_w, const int kstride_d,
	 const size_t start_idx, const size_t end_idx, 
     Dtype* data_col);
	 
template <typename Dtype>
void im2col_sk_partition_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int depth, 
	const int kernel_h, const int kernel_w, const int kernel_d,
    const int pad_h, const int pad_w, const int pad_d, 
	const int stride_h, const int stride_w, const int stride_d,
    const int kstride_h, const int kstride_w, const int kstride_d,
	const size_t start_idx, const size_t end_idx, 
    Dtype* data_col);
	
	
	
	
	

} 



 // namespace caffe

#endif  // CAFFE_UTIL_IM2COL_HPP_
