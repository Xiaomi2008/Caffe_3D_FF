// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
void PoolingLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Set the max number of top blobs before calling base Layer::SetUp.
  // If doing MAX pooling, we can optionally output an extra top Blob
  // for the mask.  Otherwise, we only have one top Blob.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX) {
    max_top_blobs_ = 2;
  } else {
    max_top_blobs_ = 1;
  }
  Layer<Dtype>::SetUp(bottom, top);
 // kernel_size_ = this->layer_param_.pooling_param().kernel_size();
 if(this->layer_param_.pooling_param().has_kernel_size()){
    kernel_h_ = kernel_w_ = kernel_d_ =this->layer_param_.pooling_param().kernel_size();
 }else
 {
	kernel_h_ = this->layer_param_.pooling_param().kernel_h();
	kernel_w_ = this->layer_param_.pooling_param().kernel_w();
	kernel_d_ = this->layer_param_.pooling_param().kernel_d();
 }
  //stride_ = this->layer_param_.pooling_param().stride();
  if(this->layer_param_.pooling_param().has_stride()){
    stride_h_ = stride_w_ = stride_d_ =this->layer_param_.pooling_param().stride();
 }else
 {
	stride_h_ = this->layer_param_.pooling_param().stride_h();
	stride_w_ = this->layer_param_.pooling_param().stride_w();
	stride_d_ = this->layer_param_.pooling_param().stride_d();
 }
 if(this->layer_param_.pooling_param().has_pad()){
	pad_h_=pad_w_=pad_d_=this->layer_param_.pooling_param().pad();
  }else{
	pad_h_=this->layer_param_.pooling_param().pad_h();
	pad_w_=this->layer_param_.pooling_param().pad_w();
	pad_d_=this->layer_param_.pooling_param().pad_d();
  }
 // pad_ = this->layer_param_.pooling_param().pad();
  if (pad_h_ != 0|| pad_w_ != 0 ||pad_d_ != 0 ) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
	CHECK_LT(pad_w_, kernel_w_);
	CHECK_LT(pad_d_, kernel_d_);
  }
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  depth_ = bottom[0]->depth();
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  pooled_depth_ = static_cast<int>(ceil(static_cast<float>(
      depth_ + 2 * pad_d_ - kernel_d_) / stride_d_)) + 1;
  if (pad_h_ ) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
		if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
		--pooled_height_;
		}
		CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
	}
	 if (pad_w_ ) {
		if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
		  --pooled_width_;
		}
		 CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);	
	   }
	  if (pad_d_ ){   
		if ((pooled_depth_ - 1) * stride_d_ >= depth_ + pad_d_) {
		  --pooled_depth_;
		}
		CHECK_LT((pooled_depth_ - 1) * stride_d_, depth_ + pad_d_);
   }	
   
  (*top)[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_, pooled_depth_);
  if (top->size() > 1) {
    (*top)[1]->ReshapeLike(*(*top)[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top->size() == 1) {
    max_idx_.reset(new Blob<int>(bottom[0]->num(), channels_,
                                 pooled_height_, pooled_width_, pooled_depth_));
	//max_idx_.Reshape(bottom[0]->num(), channels_,
    //                             pooled_height_, pooled_width_, pooled_depth_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_, pooled_depth_);
  }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
// template <typename Dtype>
// Dtype PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      // vector<Blob<Dtype>*>* top) {
  // const Dtype* bottom_data = bottom[0]->cpu_data();
  // Dtype* top_data = (*top)[0]->mutable_cpu_data();
  // const int top_count = (*top)[0]->count();
  // // We'll output the mask to top[1] if it's of size >1.
  // const bool use_top_mask = top->size() > 1;
  // int* mask = NULL;  // suppress warnings about uninitalized variables
  // Dtype* top_mask = NULL;
  // // Different pooling methods. We explicitly do the switch outside the for
  // // loop to save time, although this results in more code.
  // switch (this->layer_param_.pooling_param().pool()) {
  // case PoolingParameter_PoolMethod_MAX:
    // // Initialize
    // if (use_top_mask) {
      // top_mask = (*top)[1]->mutable_cpu_data();
      // caffe_set(top_count, Dtype(-1), top_mask);
    // } else {
      // mask = max_idx_->mutable_cpu_data();
      // caffe_set(top_count, -1, mask);
    // }
    // caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // // The main loop
    // for (int n = 0; n < bottom[0]->num(); ++n) {
      // for (int c = 0; c < channels_; ++c) {
        // for (int ph = 0; ph < pooled_height_; ++ph) {
          // for (int pw = 0; pw < pooled_width_; ++pw) {
            // int hstart = ph * stride_ - pad_;
            // int wstart = pw * stride_ - pad_;
            // int hend = min(hstart + kernel_size_, height_);
            // int wend = min(wstart + kernel_size_, width_);
            // hstart = max(hstart, 0);
            // wstart = max(wstart, 0);
            // const int pool_index = ph * pooled_width_ + pw;
            // for (int h = hstart; h < hend; ++h) {
              // for (int w = wstart; w < wend; ++w) {
                // const int index = h * width_ + w;
                // if (bottom_data[index] > top_data[pool_index]) {
                  // top_data[pool_index] = bottom_data[index];
                  // if (use_top_mask) {
                    // top_mask[pool_index] = static_cast<Dtype>(index);
                  // } else {
                    // mask[pool_index] = index;
                  // }
                // }
              // }
            // }
          // }
        // }
        // // compute offset
        // bottom_data += bottom[0]->offset(0, 1);
        // top_data += (*top)[0]->offset(0, 1);
        // if (use_top_mask) {
          // top_mask += (*top)[0]->offset(0, 1);
        // } else {
          // mask += (*top)[0]->offset(0, 1);
        // }
      // }
    // }
    // break;
  // case PoolingParameter_PoolMethod_AVE:
    // for (int i = 0; i < top_count; ++i) {
      // top_data[i] = 0;
    // }
    // // The main loop
    // for (int n = 0; n < bottom[0]->num(); ++n) {
      // for (int c = 0; c < channels_; ++c) {
        // for (int ph = 0; ph < pooled_height_; ++ph) {
          // for (int pw = 0; pw < pooled_width_; ++pw) {
            // int hstart = ph * stride_ - pad_;
            // int wstart = pw * stride_ - pad_;
            // int hend = min(hstart + kernel_size_, height_ + pad_);
            // int wend = min(wstart + kernel_size_, width_ + pad_);
            // int pool_size = (hend - hstart) * (wend - wstart);
            // hstart = max(hstart, 0);
            // wstart = max(wstart, 0);
            // hend = min(hend, height_);
            // wend = min(wend, width_);
            // for (int h = hstart; h < hend; ++h) {
              // for (int w = wstart; w < wend; ++w) {
                // top_data[ph * pooled_width_ + pw] +=
                    // bottom_data[h * width_ + w];
              // }
            // }
            // top_data[ph * pooled_width_ + pw] /= pool_size;
          // }
        // }
        // // compute offset
        // bottom_data += bottom[0]->offset(0, 1);
        // top_data += (*top)[0]->offset(0, 1);
      // }
    // }
    // break;
  // case PoolingParameter_PoolMethod_STOCHASTIC:
    // NOT_IMPLEMENTED;
    // break;
  // default:
    // LOG(FATAL) << "Unknown pooling method.";
  // }
  // return Dtype(0.);
// }

template <typename Dtype>
Dtype PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int top_count = (*top)[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top->size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = (*top)[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_->mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
		    int hstart = ph * stride_h_ - pad_h_;
			int hend = min(hstart + kernel_h_, height_);
			hstart = max(hstart, 0);
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int wstart = pw * stride_w_ - pad_w_;
            int wend = min(wstart + kernel_w_, width_);
            wstart = max(wstart, 0);
			for(int pd = 0; pd <pooled_depth_; ++pd){
			 int dstart = pd * stride_d_ - pad_d_ ;
			 int dend  = min(dstart + kernel_d_, depth_);
            const int pool_index = (ph * pooled_width_ + pw) * pooled_depth_ + pd;
				for (int h = hstart; h < hend; ++h) {
				  for (int w = wstart; w < wend; ++w) {
				    for(int d = dstart; d <dend; ++d) {
						const int index = (h * width_ + w ) * depth_ + d;
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
  case PoolingParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
			int hstart = ph * stride_h_ - pad_h_;
			int hend = min(hstart + kernel_h_, height_ + pad_h_);
			hstart = max(hstart, 0);
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int wstart = pw * stride_w_ - pad_w_;
            int wend = min(wstart + kernel_w_, width_ + pad_w_); 
			wstart = max(wstart, 0);
			 for (int pd = 0; pd < pooled_depth_; ++pd) {
				int dstart = pd * stride_d_ - pad_d_;
                int dend = min(dstart + kernel_d_, depth_ + pad_d_); 
			    dstart = max(dstart, 0);
				int pool_size = (hend - hstart) * (wend - wstart) * (dend - dstart);
				hend = min(hend, height_);
				wend = min(wend, width_);
				dend = min(dend, depth_);
				for (int h = hstart; h < hend; ++h) {
				  for (int w = wstart; w < wend; ++w) {
				    for (int d = dstart; d < dend; ++d) {
						top_data[(ph * pooled_width_ + pw) * pooled_depth_ + pd] +=
							bottom_data[(h * width_ + w) * depth_ + d];
						}
				  }
				}
				top_data[(ph * pooled_width_ + pw) * pooled_depth_ + pd] /= pool_size;
			}
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += (*top)[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  return Dtype(0.);
}



template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set((*bottom)[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_->cpu_data();
    }
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
			for (int pd = 0; pd < pooled_depth_; ++pd) {
				const int index = (ph * pooled_width_ + pw) * pooled_depth_ + pd;
				const int bottom_index =
					use_top_mask ? top_mask[index] : mask[index];
				bottom_diff[bottom_index] += top_diff[index];
				}
			}
          }
        
        bottom_diff += (*bottom)[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
			for (int pd = 0; pd < pooled_depth_; ++pd){
				int hstart = ph * stride_h_ - pad_h_;
				int wstart = pw * stride_w_ - pad_w_;
				int dstart = pd * stride_d_ - pad_d_;
				int hend = min(hstart + kernel_h_, height_ + pad_h_);
				int wend = min(wstart + kernel_w_, width_ + pad_w_);
				int dend = min(dstart + kernel_d_, depth_ + pad_d_);
				int pool_size = (hend - hstart) * (wend - wstart) * (dend - dstart);
				hstart = max(hstart, 0);
				wstart = max(wstart, 0);
				dstart = max(dstart, 0);
				hend = min(hend, height_);
				wend = min(wend, width_);
				dend = min(dend, depth_);
				for (int h = hstart; h < hend; ++h) {
				  for (int w = wstart; w < wend; ++w) {
					for (int d = dstart; d < dend; ++d) {
						bottom_diff[(h * width_ + w) * depth_ + d] +=
						  top_diff[(ph * pooled_width_ + pw) * pooled_depth_ + pd] / pool_size;
					  }
				  }
				}
			}
          }
        }
        // offset
        bottom_diff += (*bottom)[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


INSTANTIATE_CLASS(PoolingLayer);


}  // namespace caffe
