// Copyright 2014 BVLC and contributors.
//
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void SoftmaxLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  LOG(INFO)<<"start set softmax " << bottom[0]->num()<<" " <<bottom[0]->channels()<<" "<<bottom[0]->height()<<" " <<bottom[0]->width()<<" "<< bottom[0]->depth();
  
  
  outer_num_ = (size_t)bottom[0]->num();
  inner_num_ = (size_t)bottom[0]->height()*(size_t)bottom[0]->width()*(size_t)bottom[0]->depth();
  sum_multiplier_.Reshape(1, bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width(), bottom[0]->depth());
  LOG(INFO)<<"sum_multiplier_.Reshape";
  
  LOG(INFO)<<"start set softmax " <<  sum_multiplier_.num()<<" " <<sum_multiplier_.channels()<<" "<<sum_multiplier_.height()<<" " <<sum_multiplier_.width()<<" "<< sum_multiplier_.depth();
  
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
   LOG(INFO)<<"done reshap sum";
   
  // LOG(INFO)<<"";
   (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width(), bottom[0]->depth());
  LOG(INFO)<<"done reshap top";
  for (size_t i = 0; i < sum_multiplier_.count(); ++i) {
     multiplier_data[i] = 1.;
  }
  //caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  scale_.Reshape(bottom[0]->num(), 1,  bottom[0]->height(), bottom[0]->width(), bottom[0]->depth());
  
  
  LOG(INFO)<<"done set softmax";
}

template <typename Dtype>
Dtype SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  size_t num = bottom[0]->num();
  size_t channels = bottom[0]->channels();
  size_t dim = bottom[0]->count() / outer_num_;
  //size_t dim = bottom[0]->count() / bottom[0]->num();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  
  for (size_t i = 0; i < outer_num_; ++i) {
    // initialize scale_data to the first plane
    caffe_copy(inner_num_, bottom_data + i * dim, scale_data);
    for (size_t j = 0; j < channels; j++) {
      for (size_t k = 0; k < inner_num_; k++) {
        scale_data[k] = std::max(scale_data[k],
            bottom_data[i * dim + j * inner_num_ + k]);
      }
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
        1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);
    // exponentiation
    caffe_exp<Dtype>(dim, top_data, top_data);
    // sum after exp
    caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1.,
        top_data, sum_multiplier_.cpu_data(), 0., scale_data);
    // division
    for (size_t j = 0; j < channels; j++) {
      caffe_div(inner_num_, top_data, scale_data, top_data);
      top_data += inner_num_;
    }
  }
  
  
  
  // we need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  // for (int i = 0; i < num; ++i) {
    // scale_data[i] = bottom_data[i*dim];
    // for (size_t j = 0; j < dim; ++j) {
      // scale_data[i] = max(scale_data[i], bottom_data[i * dim + j]);
    // }
  // }
  // // subtraction
  // caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
    // scale_data, sum_multiplier_.cpu_data(), 1., top_data);
  // // Perform exponentiation
  // caffe_exp<Dtype>(num * dim, top_data, top_data);
  // // sum after exp
  // caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., top_data,
      // sum_multiplier_.cpu_data(), 0., scale_data);
  // // Do division
  // for (int i = 0; i < num; ++i) {
    // caffe_scal<Dtype>(dim, Dtype(1.) / scale_data[i], top_data + i * dim);
  // }
  return Dtype(0);
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  
  //int channels = top[0]->shape(softmax_axis_);
 
  size_t num = top[0]->num();
  size_t channels = top[0]->channels();
  //int dim = top[0]->count() / top[0]->num(); 
  size_t dim = top[0]->count() / outer_num_;
 // size_t dim = top[0]->count() / top[0]->num();
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  
  for (size_t i = 0; i < outer_num_; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
    for (size_t k = 0; k < inner_num_; ++k) {
      scale_data[k] = caffe_cpu_strided_dot<Dtype>(channels,
          bottom_diff + i * dim + k, inner_num_,
          top_data + i * dim + k, inner_num_);
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
        -1., sum_multiplier_.cpu_data(), scale_data, 1., bottom_diff + i * dim);
  }
  // elementwise multiplication
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
  
  
  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff
  // for (size_t i = 0; i < num; ++i) {
    // scale_data[i] = caffe_cpu_dot<Dtype>(dim, top_diff + i * dim,
        // top_data + i * dim);
  // }
  // // subtraction
  // caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      // scale_data, sum_multiplier_.cpu_data(), 1., bottom_diff);
  // // elementwise multiplication
  // caffe_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
}


INSTANTIATE_CLASS(SoftmaxLayer);


}  // namespace caffe
