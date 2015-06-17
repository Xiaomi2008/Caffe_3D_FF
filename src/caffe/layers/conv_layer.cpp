// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  LOG(INFO)<<"start setup conv layer";
  //kernel_size_ = this->layer_param_.convolution_param().kernel_size();
  if(this->layer_param_.convolution_param().has_kernel_size()){
	kernel_h_=kernel_w_=kernel_d_=this->layer_param_.convolution_param().kernel_size();
  }else{
	kernel_h_=this->layer_param_.convolution_param().kernel_h();
	kernel_w_=this->layer_param_.convolution_param().kernel_w();
	kernel_d_=this->layer_param_.convolution_param().kernel_d();
  }
  if(this->layer_param_.convolution_param().has_stride()){
	stride_h_=stride_w_=stride_d_=this->layer_param_.convolution_param().stride();
  }else{
	stride_h_=this->layer_param_.convolution_param().stride_h();
	stride_w_=this->layer_param_.convolution_param().stride_w();
	stride_d_=this->layer_param_.convolution_param().stride_d();
  }
  
  if(this->layer_param_.convolution_param().has_pad()){
	pad_h_=pad_w_=pad_d_=this->layer_param_.convolution_param().pad();
  }else{
	pad_h_=this->layer_param_.convolution_param().pad_h();
	pad_w_=this->layer_param_.convolution_param().pad_w();
	pad_d_=this->layer_param_.convolution_param().pad_d();
  }
  
  //stride_ = this->layer_param_.convolution_param().stride();
  group_ = this->layer_param_.convolution_param().group();
 // pad_ = this->layer_param_.convolution_param().pad();
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  depth_ = bottom[0]->depth();
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
        << "Inputs must have same width.";
	CHECK_EQ(depth_, bottom[bottom_id]->depth())
        << "Inputs must have same depth.";
  }
  num_output_ = this->layer_param_.convolution_param().num_output();
  LOG(INFO)<< "num of output from param = " <<num_output_;
  CHECK_GT(num_output_, 0);
  CHECK_EQ(channels_ % group_, 0);
  // The im2col result buffer would only hold one image at a time to avoid
  // overly large memory usage.
  int height_out = (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  int width_out = (width_ + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
  int depth_out = (depth_ + 2 * pad_d_ - kernel_d_) / stride_d_ + 1;
  col_buffer_.Reshape(
      1, channels_ * kernel_h_ * kernel_w_* kernel_d_, height_out, width_out, depth_out);
	  
   LOG(INFO)<<"Done col_buffer_.Reshape";
  // Set the parameters
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  // Figure out the dimensions for individual gemms.
  M_ = num_output_ / group_;
  //K_ = channels_ * kernel_size_ * kernel_size_ / group_;
  //N_ = height_out * width_out;
  K_ = channels_ * kernel_h_ * kernel_w_ * kernel_d_ / group_;
  N_ = height_out * width_out * depth_out;
  for (int top_id = 0; top_id < top->size(); ++top_id) {
    (*top)[top_id]->Reshape(num_, num_output_, height_out, width_out, depth_out);
  }
  LOG(INFO)<<"Done (*top)[top_id]->Reshape";
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
	
	LOG(INFO)<<"start seting up weight";
    // Intialize the weight
    this->blobs_[0].reset(new Blob<Dtype>(
        num_output_, channels_ / group_, kernel_h_, kernel_w_, kernel_d_));
    // fill the weights
	
	LOG(INFO)<<"num_output = "<<num_output_<<" chanels /group = " << channels_ <<"/" <<group_<<" =" <<channels_ / group_;
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    LOG(INFO)<<"weight filler start  fill weight";
    weight_filler->Fill(this->blobs_[0].get());
	
	 LOG(INFO)<<"done setup weight";
    // If necessary, initialize and fill the bias term
    if (bias_term_) {
	   LOG(INFO)<<"convolution layer bias setting: there is/are " << num_output_ << " of bias";
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, 1, num_output_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Set up the bias filler
  if (bias_term_) {
    bias_multiplier_.reset(new SyncedMemory(N_ * sizeof(Dtype)));
    Dtype* bias_multiplier_data =
        reinterpret_cast<Dtype*>(bias_multiplier_->mutable_cpu_data());
    for (int i = 0; i < N_; ++i) {
        bias_multiplier_data[i] = 1.;
    }
  }
  
  LOG(INFO)<<"Done setup conv layer";
}


template <typename Dtype>
Dtype ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = (*top)[i]->mutable_cpu_data();
    Dtype* col_data = col_buffer_.mutable_cpu_data();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    int weight_offset = M_ * K_;
    int col_offset = K_ * N_;
    int top_offset = M_ * N_;
    for (int n = 0; n < num_; ++n) {
      // First, im2col
      im2col_cpu(bottom_data + bottom[i]->offset(n), channels_, height_,
                        width_, depth_, kernel_h_,  kernel_w_,  kernel_d_, 
						pad_h_, pad_w_, pad_d_, stride_h_, stride_w_, stride_d_,
						col_data);
      // Second, innerproduct with groups
      for (int g = 0; g < group_; ++g) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
          (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
          (Dtype)0., top_data + (*top)[i]->offset(n) + top_offset * g);
      }
      // third, add bias
      if (bias_term_) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
            N_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
            reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
            (Dtype)1., top_data + (*top)[i]->offset(n));
      }
	  
	 
	  
		// if(this->layer_param_.name()=="fc"){
					    // const Dtype* bias_v =this->blobs_[1]->cpu_data();
						// for(size_t t=0;t<this->blobs_[1]->count();t++){
							// d_test+=top_data[t];
							// if(top_data[t]!=0 && name_ =="fc_1"){LOG(INFO)<<top_data[t];}
							// if(isnan(top_data[t])&&name_ =="fc_1"){
							// LOG(INFO)<<"bias ["<< t<<"]="<<bias_v[t]<<"out of " <<this->blobs_[1]->count();							
							// sleep(100);
							// }
						// }
	  
	  
    }
  }
  
  return Dtype(0.);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  Dtype* bias_diff = NULL;
  if (bias_term_) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    caffe_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }
  const int weight_offset = M_ * K_;
  const int col_offset = K_ * N_;
  const int top_offset = M_ * N_;
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = (*bottom)[i]->cpu_data();
    Dtype* bottom_diff = (*bottom)[i]->mutable_cpu_diff();
    Dtype* col_data = col_buffer_.mutable_cpu_data();
    Dtype* col_diff = col_buffer_.mutable_cpu_diff();

    // Bias gradient, if necessary.
    if (bias_term_) {
      for (int n = 0; n < num_; ++n) {
        caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
            1., top_diff + top[0]->offset(n),
            static_cast<const Dtype*>(bias_multiplier_->cpu_data()), 1.,
            bias_diff);
      }
    }
    for (int n = 0; n < num_; ++n) {
      // Since we saved memory in the forward pass by not storing all col data,
      // we will need to recompute them.
      im2col_cpu(bottom_data + (*bottom)[i]->offset(n), channels_, height_,
                 width_, depth_, kernel_h_,kernel_w_,kernel_d_, 
				 pad_h_, pad_w_, pad_d_,
				 stride_h_,stride_w_,stride_d_,
				 col_data);
      // gradient w.r.t. weight. Note that we will accumulate diffs.
      for (int g = 0; g < group_; ++g) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
          (Dtype)1., top_diff + top[i]->offset(n) + top_offset * g,
          col_data + col_offset * g, (Dtype)1.,
          weight_diff + weight_offset * g);
      }
      // gradient w.r.t. bottom data, if necessary
      if (propagate_down[i]) {
        for (int g = 0; g < group_; ++g) {
          caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
            (Dtype)1., weight + weight_offset * g,
            top_diff + top[i]->offset(n) + top_offset * g,
            (Dtype)0., col_diff + col_offset * g);
        }
        // col2im back to the data
        col2im_cpu(col_diff, channels_, height_, width_, depth_, kernel_h_,kernel_w_,kernel_d_, 
					pad_h_, pad_w_, pad_d_, stride_h_, stride_w_, stride_d_,
					bottom_diff + (*bottom)[i]->offset(n));
      }
    }
  }
}

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
