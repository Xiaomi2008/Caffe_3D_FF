// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
__global__ void kernel_channel_max(const size_t num, const size_t channels,
    const size_t spatial_dim, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    size_t n = index / spatial_dim;
    size_t s = index % spatial_dim;
    Dtype maxval = -FLT_MAX;
    for (size_t c = 0; c < channels; ++c) {
      maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}

template <typename Dtype>
__global__ void kernel_channel_subtract(const size_t count,
    const size_t num, const size_t channels,
    const size_t spatial_dim, const Dtype* channel_max, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    size_t n = index / channels / spatial_dim;
    size_t s = index % spatial_dim;
    data[index] -= channel_max[n * spatial_dim + s];
  }
}


template <typename Dtype>
__global__ void kernel_channel_sum(const size_t num, const size_t channels,
    const size_t spatial_dim, const Dtype* data, Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    size_t n = index / spatial_dim;
    size_t s = index % spatial_dim;
    Dtype sum = 0;
    for (size_t c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

template <typename Dtype>
__global__ void kernel_channel_div(const size_t count,
    const size_t num, const size_t channels,
    const size_t spatial_dim, const Dtype* channel_sum, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    size_t n = index / channels / spatial_dim;
    size_t s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_channel_dot(const size_t num, const size_t channels,
    const size_t spatial_dim, const Dtype* data_1, const Dtype* data_2,
    Dtype* channel_dot) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype dot = 0;
    for (size_t c = 0; c < channels; ++c) {
      dot += (data_1[(n * channels + c) * spatial_dim + s]
          * data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}




template <typename Dtype>
__global__ void kernel_get_max(const int num, const int dim,
    const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num) {
    Dtype maxval = -FLT_MAX;
    for (int i = 0; i < dim; ++i) {
      maxval = max(data[index * dim + i], maxval);
    }
    out[index] = maxval;
  }
}

template <typename Dtype>
__global__ void kernel_softmax_div(const int num, const int dim,
    const Dtype* scale, Dtype* data) {
  CUDA_KERNEL_LOOP(index, num * dim) {
    int n = index / dim;
    data[index] /= scale[n];
  }
}

template <typename Dtype>
__global__ void kernel_exp(const int num, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num) {
    out[index] = exp(data[index]);
  }
}

template <typename Dtype>
Dtype SoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
	//LOG(INFO)<<"SoftmaxLayer forward_pgu ";
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  
  size_t count = bottom[0]->count();
  size_t channels = (*top)[0]->channels();
  size_t num = bottom[0]->num();
  size_t dim = bottom[0]->count() / bottom[0]->num();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  
  kernel_channel_max<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_, top_data,
      scale_data);
	  
  // subtract
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
      scale_data, top_data);
  // exponentiate
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_data, top_data);
  // sum after exp
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_sum<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_, top_data,
      scale_data);
  // divide
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
      scale_data, top_data);
	  
  
  
  // // we need to subtract the max to avoid numerical issues, compute the exp,
  // // and then normalize.
  // // Compute max
  // // NOLINT_NEXT_LINE(whitespace/operators)
  // kernel_get_max<Dtype><<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS>>>(
      // num, dim, bottom_data, scale_data);
  // // subtraction
  // caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      // scale_data, sum_multiplier_.gpu_data(), 1., top_data);
  // // Perform exponentiation
  // // NOLINT_NEXT_LINE(whitespace/operators)
  // kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(num * dim), CAFFE_CUDA_NUM_THREADS>>>(
      // num * dim, top_data, top_data);
  // // sum after exp
  // caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., top_data,
      // sum_multiplier_.gpu_data(), 0., scale_data);
  // // Do division
  // // NOLINT_NEXT_LINE(whitespace/operators)
  // kernel_softmax_div<Dtype><<<CAFFE_GET_BLOCKS(num * dim),
                              // CAFFE_CUDA_NUM_THREADS>>>(
      // num, dim, scale_data, top_data);
  //LOG(INFO)<<"Done Softmax GPU...";
  return Dtype(0);
}

// TODO(Yangqing): implement the GPU version of softmax.
template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
	LOG(INFO)<<"SoftmaxLayer backward_pgu ";
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  Dtype* scale_data = scale_.mutable_gpu_data();
  size_t num = top[0]->num();
  size_t count = top[0]->count();
  size_t channels = top[0]->channels();
  //int dim = top[0]->count() / top[0]->num();
  size_t dim = top[0]->count() / top[0]->num();
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  
  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff.
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_dot<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_,
      top_diff, top_data, scale_data);
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
      scale_data, bottom_diff);
  // elementwise multiplication
  caffe_gpu_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
  
  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff
  // cuda dot returns the result to cpu, so we temporarily change the pointer
  // mode
  // CUBLAS_CHECK(cublasSetPointerMode(Caffe::cublas_handle(),
      // CUBLAS_POINTER_MODE_DEVICE));
  // Dtype* scale_data = scale_.mutable_gpu_data();
  // for (int i = 0; i < num; ++i) {
    // caffe_gpu_dot<Dtype>(dim, top_diff + i * dim,
        // top_data + i * dim, scale_data + i);
  // }
  // CUBLAS_CHECK(cublasSetPointerMode(Caffe::cublas_handle(),
      // CUBLAS_POINTER_MODE_HOST));
  // // subtraction
  // caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      // scale_.gpu_data(), sum_multiplier_.gpu_data(), 1., bottom_diff);
  // // elementwise multiplication
  // caffe_gpu_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
  LOG(INFO)<<"done SoftmaxLayer backward_gpu ";
}

INSTANTIATE_CLASS(SoftmaxLayer);


}  // namespace caffe
