#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


// (const int nthreads, const Dtype* bottom_data,
    // const int num, const int channels, const int height,
    // const int width, const int pooled_height, const int pooled_width,
    // const int kernel_h, const int kernel_w, const int ext_kernel_h, const int ext_kernel_w,
    // const int stride_h, const int stride_w, const int kstride_h, const int kstride_w, 
    // const int pad_h, const int pad_w, Dtype* top_data,
    // int* mask, Dtype* top_mask) 


namespace caffe {


// __global__ void MaxPoolForward(const int nthreads, const Dtype* bottom_data,
    // const int num, const int channels, const int height, const int width, const int depth, 
	// const int pooled_height, const int pooled_width, const int pooled_depth, 
	// const int kernel_h, const int kernel_w, const int kernel_d,
    // const int ext_kernel_h, const int ext_kernel_w, const int ext_kernel_d,	
	// const int stride_h, const int stride_w, const int stride_d, 
	// const int kstride_h, const int kstride_w, const int kstride_d,
	// const int pad_h, const int pad_w, const int pad_d, Dtype* top_data,
    // int* mask, Dtype* top_mask)
template <typename Dtype>
__global__ void MaxPoolForward(const size_t nthreads, const Dtype* bottom_data,
    const size_t num, const size_t channels, const size_t height, const size_t width, const size_t depth, 
	const size_t pooled_height, const size_t pooled_width, const size_t pooled_depth, 
	const size_t kernel_h, const size_t kernel_w, const size_t kernel_d,
    const size_t ext_kernel_h, const size_t ext_kernel_w, const size_t ext_kernel_d,	
	const size_t stride_h, const size_t stride_w, const size_t stride_d, 
	const size_t kstride_h, const size_t kstride_w, const size_t kstride_d,
	const size_t pad_h, const size_t pad_w, const size_t pad_d, Dtype* top_data,
    int* mask, Dtype* top_mask)
	{
	typedef unsigned long long ulong;
  CUDA_KERNEL_LOOP(index, nthreads) {
	ulong pd = index % pooled_depth;
	ulong pw = (index / pooled_depth) % pooled_width;
    ulong ph = (index / pooled_depth / pooled_width ) % pooled_height;
    ulong c = (index / pooled_depth / pooled_width / pooled_height) % channels;
    ulong n = index / pooled_depth / pooled_width / pooled_height / channels;
    ulong hstart = ph * stride_h - pad_h;
    ulong wstart = pw * stride_w - pad_w;
	ulong dstart = pd * stride_d - pad_d;
    ulong hend = min(hstart + ext_kernel_h, (ulong)height);
    ulong wend = min(wstart + ext_kernel_w, (ulong)width);
	ulong dend = min(dstart + ext_kernel_d, (ulong)depth);
    hstart = max(hstart, (ulong)0);
    wstart = max(wstart, (ulong)0);
	dstart = max(dstart, (ulong)0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    bottom_data += (n * channels + c) * height * width * depth;
    for (size_t h = hstart; h < hend; h+=kstride_h ) {
      for (size_t w = wstart; w < wend; w+=kstride_w) {
	    for (size_t d = dstart; d < dend; d+=kstride_d){
		const size_t m_idx =  (h * width + w) * depth +d;
        if (bottom_data[m_idx] > maxval) {
          //maxidx = (h * width + w) * depth + d;
		  maxidx = static_cast<int>(m_idx);
          maxval = bottom_data[maxidx];
		  }
        }
      }
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}
  

template <typename Dtype>
__global__ void MaxPoolBackward_SK(const int nthreads, const Dtype* top_diff,
    const int* mask, const Dtype* top_mask, const int num, const int channels,
    const int height, const int width, const int depth, const int pooled_height,
    const int pooled_width, const int pooled_depth, 
	const int kernel_h, const int kernel_w, const int kernel_d, 
	const int ext_kernel_h, const int ext_kernel_w, const int ext_kernel_d, 
	const int stride_h, const int stride_w, const int stride_d,	
    const int pad_h, const int pad_w, const int pad_d, 
	const int k_stride_h, const int k_stride_w, const int k_stride_d,	
	Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
	int d = index % depth;
    int w = (index / depth) % width;
    int h = (index / depth / width) % height;
    int c = (index / depth / width / height) % channels;
    int n = index / depth / width / height / channels;
    // int phstart =
        // (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    // int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    // int pwstart =
        // (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    // int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
	
	// int pdstart =
        // (d + pad_d < kernel_d) ? 0 : (d + pad_d - kernel_d) / stride_d + 1;
    // int pdend = min((d + pad_d) / stride_d + 1, pooled_depth);
	
	
	int phstart =
        (h + pad_h < ext_kernel_h) ? 0 : (h + pad_h - ext_kernel_h) / stride_h + 1;
    int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    int pwstart =
        (w + pad_w < ext_kernel_w) ? 0 : (w + pad_w - ext_kernel_w) / stride_w + 1;
    int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
	
	int pdstart =
        (d + pad_d < ext_kernel_d) ? 0 : (d + pad_d - ext_kernel_d) / stride_d + 1;
    int pdend = min((d + pad_d) / stride_d + 1, pooled_depth);
	
    Dtype gradient = 0;
    int offset = (n * channels + c) * pooled_height * pooled_width * pooled_depth;
    top_diff += offset;
    if (mask) {
      mask += offset;
      // for (int ph = phstart; ph < phend; ++ph) {
        // for (int pw = pwstart; pw < pwend; ++pw) {
		  // for (int pd = pdstart; pd < pdend; ++pd) {
	   for (int ph = phstart; ph < phend; ph+=k_stride_h) {
        for (int pw = pwstart; pw < pwend; pw+=k_stride_w) {
		  for (int pd = pdstart; pd < pdend; pd+=k_stride_d) {
          if (mask[(ph * pooled_width + pw) * pooled_depth + pd] == (h * width + w) * depth +d) {
            gradient += top_diff[(ph * pooled_width + pw) * pooled_depth + pd];
			}
		  }
        }
      }
    } else {
      top_mask += offset;
      // for (int ph = phstart; ph < phend; ++ph) {
        // for (int pw = pwstart; pw < pwend; ++pw) {
			// for (int pd = pdstart; pd < pdend; ++pd){
	  for (int ph = phstart; ph < phend; ph+=k_stride_h) {
        for (int pw = pwstart; pw < pwend; pw+=k_stride_w) {
		  for (int pd = pdstart; pd < pdend; pd+=k_stride_d) {
			  if (top_mask[(ph * pooled_width + pw) * pooled_depth + pd] == (h * width + w) * depth +d) {
				gradient += top_diff[(ph * pooled_width + pw) * pooled_depth + pd];
			  }
		  }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}
  
  
  

template <typename Dtype>
__global__ void AvePoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height, const int width, const int depth, 
	const int pooled_height, const int pooled_width, const int pooled_depth, 
	const int kernel_h, const int kernel_w, const int kernel_d,
    const int ext_kernel_h, const int ext_kernel_w, const int ext_kernel_d,	
	const int stride_h, const int stride_w, const int stride_d, 
	const int kstride_h, const int kstride_w, const int kstride_d,
	const int pad_h, const int pad_w, const int pad_d, Dtype* top_data)
	{
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pd = index % pooled_depth;
	int pw = (index / pooled_depth) % pooled_width;
    int ph = (index / pooled_depth / pooled_width ) % pooled_height;
    int c = (index / pooled_depth / pooled_width / pooled_height) % channels;
    int n = index / pooled_depth / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
	int dstart = pd * stride_d - pad_d;
    int hend = min(hstart + ext_kernel_h, height + pad_h);
    int wend = min(wstart + ext_kernel_w, width + pad_w);
	int dend = min(dstart + ext_kernel_d, depth + pad_d);
	int pool_size = (hend - hstart) * (wend - wstart) * (dend - dstart);
	hstart = max(hstart, 0);
    wstart = max(wstart, 0);
	dstart = max(dstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
	dend = min(dend, depth);
    Dtype aveval = 0;
    //bottom_data += (n * channels + c) * height * width;
	bottom_data += (n * channels + c) * height * width * depth;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
		for (int d = dstart; d < dend; ++d){
			aveval += bottom_data[(h * width + w) * depth + d];
		}
      }
    }
    top_data[index] = aveval / pool_size;
  }
}

template <typename Dtype>
__global__ void StoPoolForwardTrain(const int nthreads,
    const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int ext_kernel_h, const int ext_kernel_w,
    const int stride_h, const int stride_w, const int kstride_h, const int kstride_w,
    Dtype* rand_idx, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h;
    int hend = min(hstart + ext_kernel_h, height);
    int wstart = pw * stride_w;
    int wend = min(wstart + ext_kernel_w, width);
    Dtype cumsum = 0.;
    bottom_data += (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; h += kstride_h) {
      for (int w = wstart; w < wend; w += kstride_w) {
        cumsum += bottom_data[h * width + w];
      }
    }
    float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int h = hstart; h < hend; h += kstride_h) {
      for (int w = wstart; w < wend; w += kstride_w) {
        cumsum += bottom_data[h * width + w];
        if (cumsum >= thres) {
          rand_idx[index] = ((n * channels + c) * height + h) * width + w;
          top_data[index] = bottom_data[h * width + w];
          return;
        }
      }
    }
	
  }
}


template <typename Dtype>
__global__ void StoPoolForwardTest(const int nthreads,
    const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w,  const int ext_kernel_h, const int ext_kernel_w,
    const int stride_h, const int stride_w, const int kstride_h, const int kstride_w,
    Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h;
    int hend = min(hstart + ext_kernel_h, height);
    int wstart = pw * stride_w;
    int wend = min(wstart + ext_kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    Dtype cumsum = FLT_MIN;
    Dtype cumvalues = 0.;
    bottom_data += (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; h += kstride_h) {
      for (int w = wstart; w < wend; w += kstride_w) {
        cumsum += bottom_data[h * width + w];
        cumvalues += bottom_data[h * width + w] * bottom_data[h * width + w];
      }
    }
    top_data[index] = cumvalues / cumsum;
  }
}


template <typename Dtype>
Dtype PoolingSKLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  size_t count = (*top)[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top->size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;

  size_t ext_kernel_h = (kernel_h_ - 1) * kstride_h_ + 1;
  size_t ext_kernel_w = (kernel_w_ - 1) * kstride_w_ + 1;
  size_t ext_kernel_d = (kernel_d_ - 1) * kstride_d_ + 1;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = (*top)[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, depth_, pooled_height_, pooled_width_, pooled_depth_, kernel_h_,
        kernel_w_, kernel_d_, ext_kernel_h, ext_kernel_w, ext_kernel_d,
        stride_h_, stride_w_, stride_d_, kstride_h_, kstride_w_, kstride_d_,
        pad_h_, pad_w_, pad_d_, top_data,
        mask, top_mask);
	//bottom[0]->release_data();
	//bottom[0]->release_cpu_data();
	
	
	if (Caffe::phase() == Caffe::TEST){
		bottom[0]->release_all_data();
		 if (use_top_mask) {
		  (*top)[1]->release_all_data();
		} else {
		  max_idx_.release_all_data();
		}
	  }
	 
   // Dtype* top_data = (*top)[i]->mutable_gpu_data();
    //col_buffer_.release_data();
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, depth_, pooled_height_, pooled_width_, pooled_depth_, kernel_h_,
        kernel_w_, kernel_d_, ext_kernel_h, ext_kernel_w, ext_kernel_d,
        stride_h_, stride_w_, stride_d_, kstride_h_, kstride_w_, kstride_d_,
        pad_h_, pad_w_, pad_d_, top_data);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    if (Caffe::phase() == Caffe::TRAIN) {
      // We need to create the random index as well.
      caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                            rand_idx_.mutable_gpu_data());
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTrain<Dtype><<<CAFFE_GET_BLOCKS(count),
                                   CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, ext_kernel_h, ext_kernel_w,
          stride_h_, stride_w_, kstride_h_, kstride_w_,
          rand_idx_.mutable_gpu_data(), top_data);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTest<Dtype><<<CAFFE_GET_BLOCKS(count),
                                  CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, ext_kernel_h, ext_kernel_w,
          stride_h_, stride_w_, kstride_h_, kstride_w_, top_data);
    }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
  //return;
  return Dtype(0.);
}

template <typename Dtype>
void PoolingSKLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
	  
	   if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  const int count = (*bottom)[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;	
  int ext_kernel_h = (kernel_h_ - 1) * kstride_h_ + 1;
  int ext_kernel_w = (kernel_w_ - 1) * kstride_w_ + 1;
  int ext_kernel_d = (kernel_d_ - 1) * kstride_d_ + 1;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_.gpu_data();
    }
	

	
	
	
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolBackward_SK<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, top[0]->num(), channels_,
        height_, width_, depth_, pooled_height_, pooled_width_, pooled_depth_,
        kernel_h_,kernel_w_,kernel_d_, 
		ext_kernel_h, ext_kernel_w, ext_kernel_d,
		stride_h_,stride_w_,stride_d_, pad_h_, pad_w_, pad_d_,
        kstride_h_, kstride_w_, kstride_d_,
		bottom_diff);
    break;
	 default:
    LOG(FATAL) << "Only support max pooling method for backward propagation...";
  }
  CUDA_POST_KERNEL_CHECK;
  return;
}


INSTANTIATE_CLASS(PoolingSKLayer);


}  // namespace caffe
