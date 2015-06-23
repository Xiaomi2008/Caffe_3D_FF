// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void AccuracyLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  top_k_ = this->layer_param_.accuracy_param().top_k();
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_LE(top_k_, bottom[0]->count() / bottom[0]->num())
      << "top_k must be less than or equal to the number of classes.";
  
  outer_num_ = bottom[0]->num();
  inner_num_ = bottom[0]->height()*bottom[0]->width()*bottom[0]->depth();
  LOG(INFO)<<" outer_num_ = " << outer_num_<<" inner_num_ "<<inner_num_;
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count());
  // CHECK_EQ(bottom[1]->channels(), 1);
  // CHECK_EQ(bottom[1]->height(), 1);
  // CHECK_EQ(bottom[1]->width(), 1);
  // CHECK_EQ(bottom[1]->depth(), 1);
  LOG(INFO)<<"top_k_ = " <<top_k_;
  (*top)[0]->Reshape(1, 1, 1, 1, 1);
}

template <typename Dtype>
Dtype AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
   const int num_labels = bottom[0]->channels();	 
  // vector<Dtype> maxval(top_k_+1);
  // vector<int> max_id(top_k_+1);
  // int count_label=0;
  // for (int i = 0; i < num; ++i) {
    // // Top-k accuracy
    // std::fill_n(maxval.begin(), top_k_, -FLT_MAX);
    // std::fill_n(max_id.begin(), top_k_, 0);
    // for (int j = 0, k; j < dim; ++j) {
      // // insert into (reverse-)sorted top-k array
      // Dtype val = bottom_data[i * dim + j];
      // for (k = top_k_; k > 0 && maxval[k-1] < val; k--) {
        // maxval[k] = maxval[k-1];
        // max_id[k] = max_id[k-1];
      // }
      // maxval[k] = val;
      // max_id[k] = j;
    // }
    // // check if true label is in top k predictions
	// //bottom_label[i])-1 is modified by Tao Zeng 12/17/2014 for multi_label -- multi_class problem, for 0 in the label means the label dose not exist, therefore we don't count its accuracy.
	// //if (bottom_label[i]-1 >=0){
	// //	count_label++;
	// //}
	
	// //LOG(INFO)<< "label =" << static_cast<int>(bottom_label[i]);
    // for (int k = 0; k < top_k_; k++){ 
      // //if (max_id[k] == static_cast<int>(bottom_label[i])-1) {
	  // if (max_id[k] == static_cast<int>(bottom_label[i])) {
        // ++accuracy;
		// LOG(INFO)<<"equl label = " <<static_cast<int>(bottom_label[i]);
       // // break;
      // }
	// }
	  
  // //LOG(INFO) << "Non-negtive label count : " << count_label<<"  total =" <<num;
  // }
  
  
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);
  int count = 0;
  for (int i = 0; i < outer_num_; ++i) {
    //  LOG(INFO)<<" ";
   // LOG(INFO)<<"label image "<<i<<"..............";
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);
      //if (has_ignore_label_ && label_value == ignore_label_) {
      //  continue;
      //}
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);
      // Top-k accuracy
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < num_labels; ++k) {
        bottom_data_vector.push_back(std::make_pair(
            bottom_data[i * dim + k * inner_num_ + j], k));
      }
      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
      // check if true label is in top k predictions
      for (int k = 0; k < top_k_; k++) {
        if (bottom_data_vector[k].second == label_value) {
          ++accuracy;
		  LOG(INFO)<<"equl label = " <<label_value;
          break;
        }
      }
      ++count;
    }
  }

  
  
  
  
  //(*top)[0]->mutable_cpu_data()[0] = 0.011;
  (*top)[0]->mutable_cpu_data()[0] = accuracy / count;
  // Accuracy layer should not be used as a loss function.
  return Dtype(0);
}

INSTANTIATE_CLASS(AccuracyLayer);

}  // namespace caffe
