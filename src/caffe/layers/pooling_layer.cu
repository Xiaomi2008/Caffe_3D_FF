// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int depth, const int pooled_height, const int pooled_width, const int pooled_depth, const int kernel_h, const int kernel_w, const int kernel_d, const int stride_h, const int stride_w, const int stride_d, const int pad_h, const int pad_w, const int pad_d, Dtype* top_data,
    int* mask, Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pd = index % pooled_depth;
	int pw = (index / pooled_depth) % pooled_width;
    int ph = (index / pooled_depth / pooled_width ) % pooled_height;
    int c = (index / pooled_depth / pooled_width / pooled_height) % channels;
    int n = index / pooled_depth / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
	int dstart = pd * stride_d - pad_d;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
	int dend = min(dstart + kernel_d, depth);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
	dstart = max(dstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    bottom_data += (n * channels + c) * height * width * depth;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
	    for (int d = dstart; d < dend; ++d){
		const int m_idx =  (h * width + w) * depth +d;
        if (bottom_data[m_idx] > maxval) {
          //maxidx = (h * width + w) * depth + d;
		  maxidx = m_idx;
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
__global__ void AvePoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int depth, const int pooled_height, const int pooled_width, const int pooled_depth, 
	const int kernel_h, const int kernel_w, const int kernel_d, 
	const int stride_h, const int stride_w, const int stride_d, 
	const int pad_h, const int pad_w, const int pad_d, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pd = index % pooled_depth;
	int pw = (index / pooled_depth) % pooled_width;
    int ph = (index / pooled_depth / pooled_width ) % pooled_height;
    int c = (index / pooled_depth / pooled_width / pooled_height) % channels;
    int n = index / pooled_depth / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
	int dstart = pd * stride_d - pad_d;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
	int dend = min(dstart + kernel_d, depth + pad_d);
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
    const int kernel_size, const int stride, Dtype* rand_idx, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride;
    int hend = min(hstart + kernel_size, height);
    int wstart = pw * stride;
    int wend = min(wstart + kernel_size, width);
    Dtype cumsum = 0.;
    bottom_data += (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[h * width + w];
      }
    }
    float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
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
    const int kernel_h, const int kernel_w, const int kernel_d, 
	const int stride_h, const int stride_w, const int stride_d, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h;
    int hend = min(hstart + kernel_h, height);
    int wstart = pw * stride_w;
    int wend = min(wstart + kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    Dtype cumsum = FLT_MIN;
    Dtype cumvalues = 0.;
    bottom_data += (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[h * width + w];
        cumvalues += bottom_data[h * width + w] * bottom_data[h * width + w];
      }
    }
    top_data[index] = cumvalues / cumsum;
  }
}


template <typename Dtype>
Dtype PoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  int count = (*top)[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top->size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = (*top)[1]->mutable_gpu_data();
    } else {
      mask = max_idx_->mutable_gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, depth_, pooled_height_, pooled_width_, pooled_depth_ , kernel_h_,kernel_w_,kernel_d_, stride_h_,stride_w_,stride_d_,  pad_h_, pad_w_, pad_d_, top_data, mask, top_mask);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, depth_, pooled_height_, pooled_width_, pooled_depth_, kernel_h_,kernel_w_,kernel_d_, stride_h_,stride_w_,stride_d_,  pad_h_, pad_w_, pad_d_, top_data);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    // this part is not yet implemented in 3D verion.
    // if (Caffe::phase() == Caffe::TRAIN) {
      // // We need to create the random index as well.
      // caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                            // rand_idx_.mutable_gpu_data());
      // // NOLINT_NEXT_LINE(whitespace/operators)
      // StoPoolForwardTrain<Dtype><<<CAFFE_GET_BLOCKS(count),
                                   // CAFFE_CUDA_NUM_THREADS>>>(
          // count, bottom_data, bottom[0]->num(), channels_,
          // height_, width_, pooled_height_, pooled_width_, kernel_size_, stride_,
          // rand_idx_.mutable_gpu_data(), top_data);
    // } else {
      // // NOLINT_NEXT_LINE(whitespace/operators)
      // StoPoolForwardTest<Dtype><<<CAFFE_GET_BLOCKS(count),
                                  // CAFFE_CUDA_NUM_THREADS>>>(
          // count, bottom_data, bottom[0]->num(), channels_,
          // height_, width_, pooled_height_, pooled_width_, kernel_size_, stride_,
          // top_data);
    // }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
  return Dtype(0.);
}


template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* mask, const Dtype* top_mask, const int num, const int channels,
    const int height, const int width, const int depth, const int pooled_height,
    const int pooled_width, const int pooled_depth, 
	const int kernel_h, const int kernel_w, const int kernel_d, 
	const int stride_h, const int stride_w, const int stride_d, 
    const int pad_h, const int pad_w, const int pad_d, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
	int d = index % depth;
    int w = (index / depth) % width;
    int h = (index / depth / width) % height;
    int c = (index / depth / width / height) % channels;
    int n = index / depth / width / height / channels;
    int phstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    int pwstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
	
	int pdstart =
        (d + pad_d < kernel_d) ? 0 : (d + pad_d - kernel_d) / stride_d + 1;
    int pdend = min((d + pad_d) / stride_d + 1, pooled_depth);
	
    Dtype gradient = 0;
    int offset = (n * channels + c) * pooled_height * pooled_width * pooled_depth;
    top_diff += offset;
    if (mask) {
      mask += offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
		  for (int pd = pdstart; pd < pdend; ++pd) {
          if (mask[(ph * pooled_width + pw) * pooled_depth + pd] == (h * width + w) * depth +d) {
            gradient += top_diff[(ph * pooled_width + pw) * pooled_depth + pd];
			}
		  }
        }
      }
    } else {
      top_mask += offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
			for (int pd = pdstart; pd < pdend; ++pd){
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
__global__ void AvePoolBackward(const int nthreads, const Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const int depth, const int pooled_height, const int pooled_width, const int pooled_depth,  
	const int kernel_h, const int kernel_w, const int kernel_d, 
	const int stride_h, const int stride_w, const int stride_d,
	const int pad_h, const int pad_w, const int pad_d, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
	int d = index % depth +pad_d;
    int w = (index / depth) % width + pad_w;
    int h = (index / depth / width) % height + pad_h;
    int c = (index / depth / width / height) % channels;
    int n = index / depth / width / height / channels;
    int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    int phend = min(h / stride_h + 1, pooled_height);
    int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    int pwend = min(w / stride_w + 1, pooled_width);
	int pdstart = (d < kernel_d) ? 0 : (d - kernel_d) / stride_d + 1;
    int pdend = min(d / stride_d + 1, pooled_depth);
	
    Dtype gradient = 0;
    top_diff += (n * channels + c) * pooled_height * pooled_width * pooled_depth;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
	    for (int pd = pdstart; pd < pdend; ++pd) {
			// figure out the pooling size
			int hstart = ph * stride_h - pad_h;
			int wstart = pw * stride_w - pad_w;
			int dstart = pd * stride_d - pad_d;
			int hend = min(hstart + kernel_h, height + pad_h);
			int wend = min(wstart + kernel_w, width + pad_w);
			int dend = min(dstart + kernel_d, depth + pad_d);
			int pool_size = (hend - hstart) * (wend - wstart) * (dend - dstart);
			gradient += top_diff[(ph * pooled_width + pw) * pooled_depth + pd] / pool_size;
		}
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
__global__ void StoPoolBackward(const int nthreads,
    const Dtype* rand_idx, const Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int kernel_d, 
	const int stride_h, const int stride_w, const int stride_d, Dtype* bottom_diff) {
	// note the 3d versio is not implemented for this function yetlllll
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    int phend = min(h / stride_h + 1, pooled_height);
    int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_h + 1;
    int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    rand_idx += (n * channels + c) * pooled_height * pooled_width;
    top_diff += (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff[ph * pooled_width + pw] *
            (index == static_cast<int>(rand_idx[ph * pooled_width + pw]));
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
void PoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_->gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, top[0]->num(), channels_,
        height_, width_, depth_, pooled_height_, pooled_width_, pooled_depth_,
        kernel_h_,kernel_w_,kernel_d_, stride_h_,stride_w_,stride_d_, pad_h_, pad_w_, pad_d_, bottom_diff);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->num(), channels_,
        height_, width_, depth_, pooled_height_, pooled_width_, pooled_depth_, kernel_h_,kernel_w_,kernel_d_,
		stride_h_,stride_w_,stride_d_,  pad_h_, pad_w_, pad_d_, bottom_diff);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    // NOLINT_NEXT_LINE(whitespace/operators)
    StoPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, rand_idx_.gpu_data(), top_diff,
        top[0]->num(), channels_, height_, width_, pooled_height_,
        pooled_width_, kernel_h_,kernel_w_,kernel_d_, stride_h_,stride_w_,stride_d_, bottom_diff);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_CLASS(PoolingLayer);


}  // namespace caffe
