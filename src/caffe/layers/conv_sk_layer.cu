#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
Dtype ConvolutionSKLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
 //  LOG(INFO)<<"start im2col_sk_gpu "<<name_;
  size_t ext_kernel_h = (kernel_h_ - 1) * kstride_h_ + 1;
  size_t ext_kernel_w = (kernel_w_ - 1) * kstride_w_ + 1;
  size_t ext_kernel_d = (kernel_d_ - 1) * kstride_d_ + 1;
  size_t height_out = (height_ + 2*pad_h_ - ext_kernel_h) / stride_h_ + 1;
  size_t width_out = (width_   + 2*pad_w_ - ext_kernel_w) / stride_w_ + 1;
  size_t depth_out = (depth_   + 2*pad_d_ - ext_kernel_d) / stride_d_ + 1;
  const size_t out_size=height_out * width_out * depth_out;
  //size_t n_partition = 32*channels_;
  //N_ =  out_size/ n_partition;
  N_ =  out_size/num_partition_;
  size_t n_partition =out_size/N_;
  size_t partition_len =N_;
  col_buffer_.Reshape(
      1, channels_ * kernel_h_ * kernel_w_ *kernel_d_, 1, 1, N_);
 // LOG(INFO)<<"col_buffer_.mutable_gpu_data()";
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  Blob<Dtype> temp_out_buffer;
  temp_out_buffer.Reshape(1, 1, 1, num_output_, N_);
 // LOG(INFO)<<"temp_out_buffer.mutable_gpu_data()";
  Dtype* temp_out = temp_out_buffer.mutable_gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    
	//LOG(INFO)<<"temp_out_buffer.mutable_gpu_data()";
    const Dtype* bottom_data = bottom[i]->gpu_data();
//	LOG(INFO)<<"*top)[" <<i <<"]->mutable_gpu_data()" <<" Layer =" << name_;
    Dtype* top_data = (*top)[i]->mutable_gpu_data();
//	LOG(INFO)<<"blobs_[0]->mutable_gpu_data()";
    const Dtype* weight = this->blobs_[0]->gpu_data();
	
	//M_ =1;
	//M_ = num_output_;
	M_ = num_output_ / group_;
    size_t weight_offset = M_ * K_;
    size_t col_offset = K_ * N_;
    size_t top_offset = M_ * N_;
    
    for (int n = 0; n < num_; ++n) {
       // First, im2col
	  if (n_partition>1){
	   for (size_t p=0; p<n_partition+1; p++){
			 size_t p_offset =  N_*p;
			 size_t end_index;
			 const size_t start_index =p*N_;
			 if (start_index >=out_size) break;
			 if((start_index +N_)<=out_size){
			    end_index =start_index +N_;
			 }else{
				end_index =out_size;
				N_ =end_index-start_index;
				CHECK_GT(N_,0);
				//LOG(INFO)<<"last part is not equal N_ = "<<N_;
			 }
			 CHECK_EQ(N_,end_index-start_index);
			 
			    int qt=n_partition /4;
			     
			         CHECK_GT(qt, 0);
					 // if(qt>0){
					 // if (p%qt==0){
							// LOG(INFO)<<"start im2col_sk_gpu part:  " << p <<" out of " <<n_partition <<" "<<name_ ;}
					 // }
					  im2col_sk_partition_gpu(bottom_data + bottom[i]->offset(n), channels_, height_,
						width_, depth_, kernel_h_, kernel_w_, kernel_d_, pad_h_, pad_w_, pad_d_, stride_h_, stride_w_,stride_d_, kstride_h_, kstride_w_, kstride_d_, start_index, end_index,  col_data);
					 
					 // comment this --- Do convolution for each channel implementation is very slow
					 // for (size_t ch_out=0; ch_out <num_output_; ch_out++){
						  // for (size_t g = 0; g < group_; ++g) {
							// caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
							  // (Dtype)1., weight + weight_offset * g +weight_offset * group_* ch_out, col_data + col_offset * g,
							  // (Dtype)0., top_data + (*top)[i]->offset(n) + top_offset * g +ch_out*out_size + p_offset);
						  // }
					 // }
					  // comment this --- Do convolution for all channel and copy each channels to top data is much faster :) by tao zeng May/28/2015
					 for (size_t g = 0; g < group_; ++g) {
							caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_, N_, K_,
							  (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
							  (Dtype)0., temp_out);
						  }
					for (size_t ch_out=0; ch_out <num_output_; ch_out++){
						caffe_copy(N_,temp_out+ch_out*N_,
						           top_data + (*top)[i]->offset(n)  
								   +ch_out*out_size 
								   + p_offset
								   );
					}
					 
				  
			}
	     }
		 else{
					//LOG(INFO)<<"start im2col_sk_cpu part for "<<N_;
					N_ =  out_size;
					M_ = num_output_ / group_;
					im2col_sk_gpu(bottom_data + bottom[i]->offset(n), channels_, height_,
						width_, depth_, kernel_h_, kernel_w_, kernel_d_, pad_h_, pad_w_, pad_d_, stride_h_, stride_w_,stride_d_, kstride_h_, kstride_w_, kstride_d_, col_data);
					 
					 for (int g = 0; g < group_; ++g) {
							caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
							  (Dtype)1., weight + weight_offset * g , col_data + col_offset * g,
							  (Dtype)0., top_data + (*top)[i]->offset(n) + top_offset * g );
						  }
		 
		 
		 }
			//caffe_copy(N,A,B)
			
			if (bias_term_) {
			  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
			   out_size, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
			   bias_multiplier_.gpu_data(),
			   (Dtype)1., top_data + (*top)[i]->offset(n));
			   }	
			   
		}
	    if (Caffe::phase() == Caffe::TEST){
		        //LOG(INFO)<<"bottom data released";
				bottom[i]->release_all_data();
				//bottom[i]->release_gpu_data();
				}	   		
		
   } 
  
   temp_out_buffer.release_all_data();
   if (Caffe::phase() == Caffe::TEST){ 
       //LOG(INFO)<<"data released";
       col_buffer_.release_all_data();
	   //col_buffer_.release_gpu_data();
	    bias_multiplier_.release_gpu_data();
	  // bias_multiplier_.release_all_data();
	   //this->blobs_[0]->release_gpu_data();
	   (*top)[0]->release_gpu_data();;
	}
	
  // LOG(INFO)<<"Done convolution " << name_;
  return Dtype(0.);
}

template <typename Dtype>
void ConvolutionSKLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
   //LOG(INFO)<<"this->blobs_[0]->gpu_data()";
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  //LOG(INFO)<<"this->blobs_[0]->mutable_gpu_diff()";
  caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  //LOG(INFO)<<"col_buffer_.mutable_gpu_data()";
  Dtype* col_diff = col_buffer_.mutable_gpu_diff();
  //LOG(INFO)<<"col_buffer_.mutable_gpu_diff()";
  Dtype* bias_diff = NULL;
  if (bias_term_) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }
  const int weight_offset = M_ * K_;
  const int col_offset = K_ * N_;
  const int top_offset = M_ * N_;
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    const Dtype* bottom_data = (*bottom)[i]->gpu_data();
    Dtype* bottom_diff = (*bottom)[i]->mutable_gpu_diff();
    // Bias gradient, if necessary.
    if (bias_term_) {
      for (int n = 0; n < num_; ++n) {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
            1., top_diff + top[0]->offset(n),
            static_cast<const Dtype*>(bias_multiplier_.gpu_data()),
            1., bias_diff);
      }
    }
	//LOG(INFO)<<"bias caffe_gpu_gemv passed:";
    for (int n = 0; n < num_; ++n) {
      // since we saved memory in the forward pass by not storing all col data,
      // we will need to recompute them.
	  im2col_sk_gpu(bottom_data + (*bottom)[i]->offset(n), channels_, height_,
           width_, depth_, kernel_h_, kernel_w_, kernel_d_, pad_h_, pad_w_, pad_d_, stride_h_, stride_w_,stride_d_, kstride_h_, kstride_w_,kstride_d_, col_data);
      // gradient w.r.t. weight. Note that we will accumulate diffs.
      for (int g = 0; g < group_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
          (Dtype)1., top_diff + top[i]->offset(n) + top_offset * g,
          col_data + col_offset * g, (Dtype)1.,
          weight_diff + weight_offset * g);
      }
      // gradient w.r.t. bottom data, if necessary
      if (propagate_down[i]) {
        for (int g = 0; g < group_; ++g) {
          caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
            (Dtype)1., weight + weight_offset * g,
            top_diff + top[i]->offset(n) + top_offset * g,
            (Dtype)0., col_diff + col_offset * g);
        }
        // col2im back to the data
		
		col2im_sk_gpu(col_diff, channels_, height_, width_, depth_, 
					kernel_h_,kernel_w_,kernel_d_, 
					pad_h_,	pad_w_, pad_d_, 
					stride_h_, stride_w_, stride_d_,
					kstride_h_, kstride_w_,kstride_d_,
						bottom_diff + (*bottom)[i]->offset(n));
        // col2im_gpu(col_diff, channels_, height_, width_, depth_, kernel_h_,kernel_w_,kernel_d_, 
					// pad_h_,	pad_w_, pad_d_, stride_h_, stride_w_, stride_d_,
						// bottom_diff + (*bottom)[i]->offset(n));
      }
    }
  }
}


INSTANTIATE_CLASS(ConvolutionSKLayer);

}  // namespace caffe
