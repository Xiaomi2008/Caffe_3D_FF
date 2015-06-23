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
 // LOG(INFO)<<"before  joint thread";
  JoinPrefetchThread();
  //LOG(INFO)<<"after  joint thread";
  // Copy the data
  caffe_copy(prefetch_data_->count(), prefetch_data_->cpu_data(),
      (*top)[0]->mutable_gpu_data());
  //LOG(INFO)<<"data layer data copyed";
  if (output_labels_) {
    caffe_copy(prefetch_label_->count(), prefetch_label_->cpu_data(),
        (*top)[1]->mutable_gpu_data());
  }
  
  if(output_single_label_){
	 caffe_copy(prefetch_single_label_->count(), prefetch_single_label_->cpu_data(),
               (*top)[2]->mutable_gpu_data());
  }
  
  CreatePrefetchThread();
 //  LOG(INFO)<<"after  create thread";
  return Dtype(0.);
}

INSTANTIATE_CLASS(ImageLabelDataLayer);

}  // namespace caffe