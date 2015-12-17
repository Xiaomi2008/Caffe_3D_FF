// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::FurtherSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
      CHECK_EQ(bottom[0]->num(), bottom[1]->num())<<
      "the number of input sample data and number of imput labele must be te same ...";
 if(bottom[1]->channels()*bottom[1]->height()
    *bottom[1]->width()*bottom[1]->depth()>1)
      CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "MULTI_LABEL_LOSS layer inputs must have the same count.";
  else
     extend_class_label_2_multi_task_ =true;

  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, &sigmoid_top_vec_);
  if (top->size() >= 1) {
   // sigmoid cross entropy loss (averaged across batch)
    (*top)[0]->Reshape(1, 1, 1, 1, 1);
  }
  if (top->size() == 2) {
   // softmax output
    (*top)[1]->ReshapeLike(*sigmoid_output_.get());
    (*top)[1]->ShareData(*sigmoid_output_.get());
  }
}

template <typename Dtype>
Dtype MultiLabelLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, &sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;
  int num_offset =bottom[0]->channels()*bottom[0]->height()*bottom[0]->width()*bottom[0]->depth();
  if (extend_class_label_2_multi_task_){
    // Extend single class label intended for multi-class to
    // multi-label  binary class tassk for entropy loass computaion.
    // e.g if class label =5, then it convered to -1 -1 -1 -1 1 -1 -1 ..... until numer of tasks.
      for (int i=0;i<num;++i){
        for(int j=0; j<num_offset;++j){
          int d_idx =i*num_offset+j;
          Dtype label = target[i]==j? Dtype(1):Dtype(-1);
          loss -= input_data[d_idx] * ((label > 0) - (input_data[i] >= 0)) -
              log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
        }
      }
  }
  else{
      for (int i = 0; i < count; ++i) {
        if (target[i] != 0) {
        // Update the loss only if target[i] is not 0 // 0 means this label dose not exsit
          loss -= input_data[i] * ((target[i] > 0) - (input_data[i] >= 0)) -
              log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
        }
      }
  }

  if (top->size() >= 1) {
    (*top)[0]->mutable_cpu_data()[0] = loss / num;
  }
  return loss / num;
}

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = (*bottom)[0]->count();
    const int num = (*bottom)[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = (*bottom)[1]->cpu_data();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      if (target[i] != 0) {
        bottom_diff[i] = sigmoid_output_data[i] - (target[i] > 0);
      } else {
        bottom_diff[i] = 0;
      }
    }
    // Scale down gradient
    caffe_scal(count, Dtype(1) / num, bottom_diff);
  }
}

INSTANTIATE_CLASS(MultiLabelLossLayer);


}  // namespace caffe
