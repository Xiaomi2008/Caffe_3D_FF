#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
//typedef long long unsigned int LONG
namespace caffe {

template <typename Dtype>
void ConvolutionSKLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  name_ = this->layer_param_.name();
 
  LOG(INFO)<<"start setting " <<name_;
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  num_partition_ = conv_param.num_conv_partition();
  CHECK(!conv_param.has_kernel_size() !=
      !(conv_param.has_kernel_h() && conv_param.has_kernel_w() && conv_param.has_kernel_d()))
      << "Filter size is kernel_size OR kernel_h and kernel_w and kernel_d; not both";
  CHECK(conv_param.has_kernel_size() ||
      (conv_param.has_kernel_h() && conv_param.has_kernel_w() && conv_param.has_kernel_d()))
      << "For non-square filters all kernel_h and kernel_w  kernel_d are required.";
  CHECK((!conv_param.has_pad() && conv_param.has_pad_h()
      && conv_param.has_pad_w() && conv_param.has_pad_d())
      || (!conv_param.has_pad_h() && !conv_param.has_pad_w() && !conv_param.has_pad_d()))
      << "pad is pad OR pad_h and pad_w and pad_d are required.";
  CHECK((!conv_param.has_stride() && conv_param.has_stride_h()
      && conv_param.has_stride_w() && conv_param.has_stride_d())
      || (!conv_param.has_stride_h() && !conv_param.has_stride_w() && !conv_param.has_stride_d()))
      << "Stride is stride OR stride_h and stride_w  and stride_d are required.";
  if (conv_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ =kernel_d_= conv_param.kernel_size();
  } else {
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
	kernel_d_ = conv_param.kernel_d();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_d_, 0) << "Filter dimensions cannot be zero.";
  if (!conv_param.has_pad_h()) {
    pad_h_ = pad_w_ =pad_d_= conv_param.pad();
  } else {
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
	pad_d_ = conv_param.pad_d();
  }
  CHECK_EQ(pad_h_, 0) << "pad_h_ must be 0";
  CHECK_EQ(pad_w_, 0) << "pad_w_ must be 0";
  CHECK_EQ(pad_d_, 0) << "pad_d_ must be 0";
  if (!conv_param.has_stride_h()) {
    stride_h_ = stride_w_=stride_d_ = conv_param.stride();
  } else {
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
	stride_d_ = conv_param.stride_d();
  }
  if (!conv_param.has_kstride_h()) {
    kstride_h_ = kstride_w_ =kstride_d_= conv_param.kstride();
  } else {
    kstride_h_ = conv_param.kstride_h();
    kstride_w_ = conv_param.kstride_w();
	kstride_d_ = conv_param.kstride_d();
  }
  
  group_ = this->layer_param_.convolution_param().group();
  channels_ = bottom[0]->channels();
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  CHECK_EQ(channels_ % group_, 0);
  // Set the parameters
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    this->blobs_[0].reset(new Blob<Dtype>(
        num_output_, channels_ / group_, kernel_h_, kernel_w_, kernel_d_));
	//const Dtype* weight = this->blobs_[0]->gpu_data();
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the bias term
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, 1, num_output_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  //LOG(INFO)<<"start setup conv";
  //this->param_propagate_down_.resize(this->blobs_.size(), true);
  
    num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  depth_ = bottom[0]->depth();
  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
    " convolution kernel.";
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
        << "Inputs must have same width.";
  }
  
  // Reshape blobs
  // The im2col result buffer would only hold one image at a time to avoid
  // overly large memory usage.
  
  
  int ext_kernel_h = (kernel_h_ - 1) * kstride_h_ + 1;
  int ext_kernel_w = (kernel_w_ - 1) * kstride_w_ + 1;
  int ext_kernel_d = (kernel_d_ - 1) * kstride_d_ + 1;
  int height_out = (height_ - ext_kernel_h) / stride_h_ + 1;
  int width_out = (width_ - ext_kernel_w) / stride_w_ + 1;
  int depth_out = (depth_ - ext_kernel_d) / stride_d_ + 1;
  
  
  
  LOG(INFO)<<"height " << height_ <<" "<<ext_kernel_h<<" "<<stride_h_;
  
  //LOG(INFO)<<"start col_buufer reshape";
  //col_buffer_.Reshape(
   //   1, channels_ * kernel_h_ * kernel_w_ *kernel_d_, height_out, width_out, depth_out);
  // Figure out the dimensions for individual gemms.
  M_ = num_output_ / group_;
  K_ = channels_ * kernel_h_ * kernel_w_ *kernel_d_/ group_;
  N_ = height_out * width_out * depth_out;
 LOG(INFO)<<"start top  reshape conv  for  layer "<< name_<<" : "<< num_ <<" "<<num_output_<<" "<<height_out<<" "<< width_out<<" "<<depth_out;
  for (int top_id = 0; top_id < top->size(); ++top_id) {
     CHECK_GT(num_,0);
	 CHECK_GT(num_output_,0);
	 CHECK_GT(height_out,0);
	 CHECK_GT(width_out,0);
	 CHECK_GT(depth_out,0);
    (*top)[top_id]->Reshape(num_, num_output_, height_out, width_out, depth_out);
  }
  // Set up the all ones "bias multiplier" for adding bias using blas
  if (bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1, 1, N_);
    caffe_set(N_, Dtype(1), bias_multiplier_.mutable_cpu_data());
	// bias_multiplier_.gpu_data();
	//  LOG(INFO)<<"bias_multiplier_.gpu_data()";
  }
  
  
  
  
    LOG(INFO)<<"done setup conv " <<name_;
   // if (bias_term_) {
    // bias_multiplier_.reset(new SyncedMemory(N_ * sizeof(Dtype)));
    // Dtype* bias_multiplier_data =
        // reinterpret_cast<Dtype*>(bias_multiplier_->mutable_cpu_data());
    // for (int i = 0; i < N_; ++i) {
        // bias_multiplier_data[i] = 1.;
    // }
  // }
  
  
  
}

// template<typename Dtype>
// void ConvolutionSKLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      // vector<Blob<Dtype>*>* top) {
  // num_ = bottom[0]->num();
  // height_ = bottom[0]->height();
  // width_ = bottom[0]->width();
  // depth_ = bottom[0]->depth();
  // CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
    // " convolution kernel.";
  // // TODO: generalize to handle inputs of different shapes.
  // for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    // CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    // CHECK_EQ(channels_, bottom[bottom_id]->channels())
        // << "Inputs must have same channels.";
    // CHECK_EQ(height_, bottom[bottom_id]->height())
        // << "Inputs must have same height.";
    // CHECK_EQ(width_, bottom[bottom_id]->width())
        // << "Inputs must have same width.";
	// CHECK_EQ(depth_, bottom[bottom_id]->depth())
        // << "Inputs must have same width.";
  // }
  // // The im2col result buffer would only hold one image at a time to avoid
  // // overly large memory usage.
  // int ext_kernel_h = (kernel_h_ - 1) * kstride_h_ + 1;
  // int ext_kernel_w = (kernel_w_ - 1) * kstride_w_ + 1;
  // int ext_kernel_w = (kernel_d_ - 1) * kstride_d_ + 1;
  // int height_out = (height_ - ext_kernel_h) / stride_h_ + 1;
  // int width_out = (width_ - ext_kernel_w) / stride_w_ + 1;
  // int depth_out = (depth_ - ext_kernel_d) / stride_d_ + 1;
  // col_buffer_.Reshape(
      // 1, channels_ * kernel_h_ * kernel_w_ *kernel_d_, height_out, width_out, depth_out);
  // // Figure out the dimensions for individual gemms.
  // M_ = num_output_ / group_;
  // K_ = channels_ * kernel_h_ * kernel_w_ *kernel_d_/ group_;
  // N_ = height_out * width_out * depth_out;
  // for (int top_id = 0; top_id < top->size(); ++top_id) {
    // (*top)[top_id]->Reshape(num_, num_output_, height_out, width_out, depth_out);
  // }
  // // Set up the all ones "bias multiplier" for adding bias using blas
  // if (bias_term_) {
    // bias_multiplier_.Reshape(1, 1, 1, N_);
    // caffe_set(N_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  // }
// }

template <typename Dtype>
Dtype ConvolutionSKLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
	  //size_t n_partition = 1;//128*channels_;
	  LOG(INFO)<<" ";
	  LOG(INFO)<<"start COnvolution_SK " << name_;
  size_t ext_kernel_h = (kernel_h_ - 1) * kstride_h_ + 1;
  size_t ext_kernel_w = (kernel_w_ - 1) * kstride_w_ + 1;
  size_t ext_kernel_d = (kernel_d_ - 1) * kstride_d_ + 1;
  size_t height_out = (height_ + 2*pad_h_ - ext_kernel_h) / stride_h_ + 1;
  size_t width_out = (width_   + 2*pad_w_ - ext_kernel_w) / stride_w_ + 1;
  size_t depth_out = (depth_   + 2*pad_d_ - ext_kernel_d) / stride_d_ + 1;
  const size_t out_size=height_out * width_out * depth_out;
  N_ =  out_size/num_partition_;
  size_t n_partition =out_size/N_;
  col_buffer_.Reshape(
      1, channels_ * kernel_h_ * kernel_w_ *kernel_d_, 1, 1, N_);
	  Dtype* col_data = col_buffer_.mutable_cpu_data();
  Blob<Dtype> temp_out_buffer;
  temp_out_buffer.Reshape(1, 1, 1, num_output_, N_);
  Dtype* temp_out = temp_out_buffer.mutable_cpu_data();
  
  for (int i = 0; i < bottom.size(); ++i) {
  
  
  // size_t n_partition = 1;
  LOG(INFO)<<" c =" << channels_<< " h ="<< height_out <<" w = " <<width_out <<" d = " << depth_out;

  CHECK_GE(num_partition_,1);
  
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = (*top)[i]->mutable_cpu_data();
    const Dtype* weight = this->blobs_[0]->cpu_data();
	
	M_ =1;
	//M_ = num_output_;
	//M_ = num_output_ / group_;
    size_t weight_offset = M_ * K_;
    size_t col_offset = K_ * N_;
    size_t top_offset = M_ * N_;
	
	
    for (int n = 0; n < num_; ++n) {
      // First, im2col
	  LOG(INFO)<<"stat im2col_sk_cpu";
	  LOG(INFO)<<channels_<<" "<<height_out<<" "<<
          width_out <<" "<< depth_out<<" "<< kernel_h_<<" "<< kernel_w_<<" "<<kernel_d_<<" "<< pad_h_<<" "<< pad_w_<<" "<< pad_d_<<" "<< stride_h_<<" "<< stride_w_<<" "<<stride_d_<<" "<<
          kstride_h_<<" "<< kstride_w_<<" "<< kstride_d_;
	   for (size_t p=0; p<n_partition+1; p++){
			 size_t p_offset =  N_*p;
			 size_t end_index;
			 const size_t start_index =p*N_;
			 
			 if (start_index >=out_size) break;
              //= (start_index +N_)<out_size ? start_index +N_ : out_size;
			 if((start_index +N_)<=out_size){
			    end_index =start_index +N_;
			 }else{
				//end_index =out_size+1;
				end_index =out_size;
				N_ =end_index-start_index;
				CHECK_GT(N_,0);
				//CHECK_EQ(N_,out_size);
				LOG(INFO)<<"last part is not equal N_ = "<<N_;
				//sleep(100);
			 }
			 CHECK_EQ(N_,end_index-start_index);
			 
			 int qt=n_partition /4;
			 if (n_partition>1){
			        if(qt>0){
					 if (p%qt==0){
						  LOG(INFO)<<"start im2col_sk_cpu part:  " << p <<" out of " <<n_partition;
						  }
					 }
					 
					// M_ = num_output_ / group_;
					// LOG(INFO)<<"start im2col_sk_cpu part for partial ims2col_sk";
					  im2col_sk_partition_cpu(bottom_data + bottom[i]->offset(n), channels_, height_,
						width_, depth_, kernel_h_, kernel_w_, kernel_d_, pad_h_, pad_w_, pad_d_, stride_h_, stride_w_,stride_d_, kstride_h_, kstride_w_, kstride_d_, start_index, end_index,  col_data);
						
						
						
					   
					   
					   
					 for (int ch_out=0; ch_out <num_output_; ch_out++){
					    
						  for (int g = 0; g < group_; ++g) {
							caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
							  (Dtype)1., weight + weight_offset * g +weight_offset * group_* ch_out, col_data + col_offset * g,
							  (Dtype)0., top_data + (*top)[i]->offset(n) + top_offset * g +ch_out*out_size + p_offset);
						  }
					}
					
						  
						  
					// The following intended to do faster convolution as in conv_sk_layer.cu,
					// However it is not correct for unkonw reaseon	:( ,by tao zeng May/28/2015
   //====================================================================================//					
					// for (size_t g = 0; g < group_; ++g) {
							// caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_, N_, K_,
							  // (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
							  // (Dtype)0., temp_out);
						  // }
					// for (size_t ch_out=0; ch_out <num_output_; ch_out++)
					// {
						// caffe_copy(N_,temp_out+ch_out*N_,
						           // top_data + (*top)[i]->offset(n)  
								   // +ch_out*out_size 
								   // + p_offset);
					// }	  
	//===================================================================================//				 
				}  
			 else{
					LOG(INFO)<<"start im2col_sk_cpu part for "<<N_;
					M_ = num_output_ / group_;
					im2col_sk_cpu(bottom_data + bottom[i]->offset(n), channels_, height_,
						width_, depth_, kernel_h_, kernel_w_, kernel_d_, pad_h_, pad_w_, pad_d_, stride_h_, stride_w_,stride_d_, kstride_h_, kstride_w_, kstride_d_, col_data);
					 
					 for (int g = 0; g < group_; ++g) {
							caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
							  (Dtype)1., weight + weight_offset * g , col_data + col_offset * g,
							  (Dtype)0., top_data + (*top)[i]->offset(n) + top_offset * g );
						  }
                    					  
				 }
			
				 
				 //----------------------------------
				 // col_buffer_2.release_cpu_data();	
				 //-------------------------------
			   //do  C = alpha*op( A )*op( B ) + beta*C
			 //LOG(INFO)<<"done im2col_sk_cpu " <<name_;
			  
	  }

	  if (bias_term_) {
	      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
		   out_size, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
		   bias_multiplier_.cpu_data(),
		   (Dtype)1., top_data + (*top)[i]->offset(n));

		}
	}
	
	bottom[i]->release_all_data();
	
	  LOG(INFO)<<"bottom data["<<i<<"] released ...";	
  }
  col_buffer_.release_all_data();	
  temp_out_buffer.release_all_data();
  bias_multiplier_.release_all_data();
  //double d_test=0;
   LOG(INFO)<<"end COnvolution_SK " <<name_;
 return Dtype(0.);
}

template <typename Dtype>
void ConvolutionSKLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  LOG(FATAL) << " Backward_cpu() not implemented for ConvolutionSKLayer.";
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionSKLayer);
#endif

INSTANTIATE_CLASS(ConvolutionSKLayer);

}  // namespace caffe
