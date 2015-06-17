// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
using std::min;
using std::max;

namespace caffe {

template <typename Dtype>
void PoolingSKLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Layer<Dtype>::SetUp(bottom, top);
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  CHECK(!pool_param.has_kernel_size() !=
      !(pool_param.has_kernel_h() && pool_param.has_kernel_w() && pool_param.has_kernel_d()))
      << "Filter size is kernel_size OR kernel_h and kernel_w and kernel_d; not both";
  CHECK(pool_param.has_kernel_size() ||
      (pool_param.has_kernel_h() && pool_param.has_kernel_w() && pool_param.has_kernel_d()))
      << "For non-square cube filters all kernel_h and kernel_w  kernel_d are required.";
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
      && pool_param.has_pad_w() && pool_param.has_pad_d())
      || (!pool_param.has_pad_h() && !pool_param.has_pad_w() && !pool_param.has_pad_d()))
      << "pad is pad OR pad_h and pad_w a pad_w re required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
      && pool_param.has_stride_w() && pool_param.has_stride_d())
      || (!pool_param.has_stride_h() && !pool_param.has_stride_w() && !pool_param.has_stride_d()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (pool_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = kernel_d_ = pool_param.kernel_size();
  } else {
    kernel_h_ = pool_param.kernel_h();
    kernel_w_ = pool_param.kernel_w();
	kernel_d_ = pool_param.kernel_d();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_d_, 0) << "Filter dimensions cannot be zero.";
  if (!pool_param.has_pad_h()) {
    pad_h_ = pad_w_ = pad_d_= pool_param.pad();
  } else {
    pad_h_ = pool_param.pad_h();
    pad_w_ = pool_param.pad_w();
	pad_d_ = pool_param.pad_d();
  }
  CHECK_EQ(pad_h_,0); //?
  CHECK_EQ(pad_w_,0);  //?
  CHECK_EQ(pad_d_,0);  //?
  if (!pool_param.has_stride_h()) {
    stride_h_ = stride_w_=stride_d_= pool_param.stride();
  } else {
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
	stride_d_ = pool_param.stride_d();
  }
  if (pad_h_ != 0 || pad_w_ != 0 || pad_d_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
	CHECK_LT(pad_d_, kernel_d_);
  }
  if (!pool_param.has_kstride_h()) {
    kstride_h_ = kstride_w_ = kstride_d_ = pool_param.kstride();
  } else {
    kstride_h_ = pool_param.kstride_h();
    kstride_w_ = pool_param.kstride_w();
	kstride_d_ = pool_param.kstride_d();
  }
  
  
  int ext_kernel_h = (kernel_h_ - 1) * kstride_h_ + 1;
  int ext_kernel_w = (kernel_w_ - 1) * kstride_w_ + 1;
  int ext_kernel_d = (kernel_d_ - 1) * kstride_d_ + 1;
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  depth_ = bottom[0]->depth();
  LOG(INFO)<< "depth "<< depth_<<" pad_d "<< pad_d_<<" ext_d "<<ext_kernel_d <<" stride_d " <<stride_d_;
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - ext_kernel_h) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - ext_kernel_w) / stride_w_)) + 1;
  pooled_depth_ = static_cast<int>(ceil(static_cast<float>(
      depth_ + 2 * pad_d_ - ext_kernel_d) / stride_d_)) + 1;
  LOG(INFO)<< "p_h = "<< pooled_height_ <<"p_w = " <<pooled_width_<< " p_d = " << pooled_depth_;
  (*top)[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_, pooled_depth_);
  if (top->size() > 1) {
    (*top)[1]->ReshapeLike(*(*top)[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top->size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
        pooled_width_, pooled_depth_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_, pooled_depth_);
  }
  
  
  
}

// template<typename Dtype>
// void PoolingSKLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      // vector<Blob<Dtype>*>* top) {
  // int ext_kernel_h = (kernel_h_ - 1) * kstride_h_ + 1;
  // int ext_kernel_w = (kernel_w_ - 1) * kstride_w_ + 1;
  // int ext_kernel_d = (kernel_d_ - 1) * kstride_d_ + 1;
  // channels_ = bottom[0]->channels();
  // height_ = bottom[0]->height();
  // width_ = bottom[0]->width();
  // pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      // height_ + 2 * pad_h_ - ext_kernel_h) / stride_h_)) + 1;
  // pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      // width_ + 2 * pad_w_ - ext_kernel_w) / stride_w_)) + 1;
  // pooled_depth_ = static_cast<int>(ceil(static_cast<float>(
      // depth_ + 2 * pad_d_ - ext_kernel_d) / stride_d_)) + 1;
  
  // (*top)[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      // pooled_width_, pooled_depth_);
  // if (top->size() > 1) {
    // (*top)[1]->ReshapeLike(*(*top)[0]);
  // }
  // // If max pooling, we will initialize the vector index part.
  // if (this->layer_param_.pooling_param().pool() ==
      // PoolingParameter_PoolMethod_MAX && top->size() == 1) {
    // max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
        // pooled_width_, pooled_depth_);
  // }
  // // If stochastic pooling, we will initialize the random index part.
  // if (this->layer_param_.pooling_param().pool() ==
      // PoolingParameter_PoolMethod_STOCHASTIC) {
    // rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
      // pooled_width_, pooled_depth_);
  // }
// }

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
Dtype PoolingSKLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
	LOG(INFO)<<" ";
    LOG(INFO)<<"start cpu pooling";
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  //const int top_count = (*top)[0]->count();
  const size_t top_count = (*top)[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top->size() > 1;
  //int* mask = NULL;  // suppress warnings about uninitalized variables
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  
  size_t ext_kernel_h = (kernel_h_ - 1) * kstride_h_ + 1;
  size_t ext_kernel_w = (kernel_w_ - 1) * kstride_w_ + 1;
  size_t ext_kernel_d = (kernel_d_ - 1) * kstride_d_ + 1;
  
   //int hend = min(hstart + ext_kernel_h, height + pad_h);
   // int wend = min(wstart + ext_kernel_w, width + pad_w);
//	int dend = min(dstart + ext_kernel_d, depth + pad_d);
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    
     
    // Initialize
    if (use_top_mask) {
      top_mask = (*top)[1]->mutable_cpu_data();
     // caffe_set(top_count, Dtype(-1), top_mask);
	  caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      //caffe_set(top_count, -1, mask);
	  //caffe_set(top_count, -1, mask);
	  caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    for ( size_t n = 0; n < bottom[0]->num(); ++n) {
      for ( size_t c = 0; c < channels_; ++c) {
        for ( size_t ph = 0; ph < pooled_height_; ++ph) {
		     size_t hstart = ph * stride_h_ - pad_h_;
			//int hend = min(hstart + kernel_h_, height_);
			 size_t hend = min(hstart + ext_kernel_h, height_ + pad_h_);
			hstart = max<size_t>(hstart, 0);
          for ( size_t pw = 0; pw < pooled_width_; ++pw) {
            int wstart = pw * stride_w_ - pad_w_;
            //int wend = min(wstart + kernel_w_, width_);
            wstart = max(wstart, 0);
			 size_t wend = min(wstart + ext_kernel_w, width_ + pad_w_);
			for( size_t pd = 0; pd <pooled_depth_; ++pd){
			 int dstart = pd * stride_d_ - pad_d_ ;
			// int dend  = min(dstart + kernel_d_, depth_);
			  size_t dend = min(dstart + ext_kernel_d, depth_ + pad_d_);
            //const int pool_index = (ph * pooled_width_ + pw) * pooled_depth_ + pd;
			const size_t pool_index = (ph * pooled_width_ + pw) * pooled_depth_ + pd;
				for ( size_t h = hstart; h < hend; h+=kstride_h_) {
				  for ( size_t w = wstart; w < wend; w+=kstride_w_) {
				    for( size_t d = dstart; d <dend; d+=kstride_d_) {
						//const int index = (h * width_ + w ) * depth_ + d;
						const size_t index = (h * width_ + w ) * depth_ + d;
						if (bottom_data[index] > top_data[pool_index]) {
						  top_data[pool_index] = bottom_data[index];
						  if (use_top_mask) {
							top_mask[pool_index] = static_cast<Dtype>(index);
						  } else {
							mask[pool_index] = index;
						  }
						}
					}
				  }
				}
			}
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += (*top)[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += (*top)[0]->offset(0, 1);
        } else {
          mask += (*top)[0]->offset(0, 1);
        }
      }
    }
    break;
	default:
	  LOG(FATAL) << "Pooling Forward_cpu method not implemented except for maxpooling.";
	}
	  LOG(INFO)<<"end cpu pooling";
	  LOG(INFO)<<" ";
	  
	  if (Caffe::phase() == Caffe::TEST){
		  if (use_top_mask) {
			  (*top)[1]->release_all_data();
			} else {
			  max_idx_.release_all_data();
			}
		  bottom[0]->release_all_data();
	  }
	 // LOG(INFO)<<"bottom data released ...";
  
}

template <typename Dtype>
void PoolingSKLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
	  
	  
	  
	  
  return;
}


#ifdef CPU_ONLY
STUB_GPU(PoolingSKLayer);
#endif

INSTANTIATE_CLASS(PoolingSKLayer);


}  // namespace caffe
