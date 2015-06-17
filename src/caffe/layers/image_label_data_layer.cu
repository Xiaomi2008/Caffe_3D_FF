// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using std::string;

namespace caffe {

template <typename Dtype>
Dtype ImageLabelDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  // LOG(INFO)<<"start forwarding data gpu .....";
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_->count(), prefetch_data_->cpu_data(),
      (*top)[0]->mutable_gpu_data());
  //LOG(INFO)<<"data layer data copyed";
  if (output_labels_) {
    caffe_copy(prefetch_label_->count(), prefetch_label_->cpu_data(),
        (*top)[1]->mutable_gpu_data());
		//const Dtype* top_label = (*top)[1]->mutable_gpu_data();
		//LOG(INFO)<<top_label[0];
	//LOG(INFO)<<"label layer  copied";
  }
  // Start a new prefetch thread
  
  //LOG(INFO)<<"start CreatePrefetchThread(); gpu .....";
  CreatePrefetchThread();
 // LOG(INFO)<<"done gpu forword prefettch";
  return Dtype(0.);
}

INSTANTIATE_CLASS(ImageLabelDataLayer);

}  // namespace caffe