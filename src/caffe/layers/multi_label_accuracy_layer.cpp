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
void MultiLabelAccuracyLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
    << "The data and label should have the same number of instances";
  if(bottom[0]->count()/bottom[0]->count()>1){
      CHECK_EQ(bottom[0]->channels(), bottom[1]->channels())
        << "The data and label should have the same number of channels";
  }

      CHECK_EQ(bottom[0]->height(), bottom[1]->height())
        << "The data and label should have the same height";
      CHECK_EQ(bottom[0]->width(), bottom[1]->width())
        << "The data and label should have the same width";
      CHECK_EQ(bottom[0]->depth(), bottom[1]->depth())
        << "The data and label should have the same depth";
  // Top will contain:
  // top[0] = Sensitivity or Recall (TP/P),
  // top[1] = Specificity (TN/N),
  // top[2] = Harmonic Mean of Sens and Spec, (2/(P/TP+N/TN))
  // top[3] = Precision (TP / (TP + FP))
  // top[4] = F1 Score (2 TP / (2 TP + FP + FN))
  (*top)[0]->Reshape(1, 5, 1, 1, 1);
}

template <typename Dtype>
Dtype MultiLabelAccuracyLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  Dtype true_positive = 0;
  Dtype false_positive = 0;
  Dtype true_negative = 0;
  Dtype false_negative = 0;
  int count_pos = 0;
  int count_neg = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())<<
  "the number of input sample data and number of imput labele must be te same ...";
  if(bottom[1]->count()/bottom[1]->num()>1)
    CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
    "MULTI_LABEL_LOSS layer inputs must have the same count.";
  else
   extend_class_label_2_multi_task_ =true;

  // Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int count = bottom[0]->count();
  const int num = bottom[0]->num();

  int num_offset =bottom[0]->channels()*bottom[0]->height()*bottom[0]->width()*bottom[0]->depth();
  if (extend_class_label_2_multi_task_){
    // Extend single class label intended for multi-class to
    // multi-label  binary class tassk for entropy loass computaion.
    // e.g if class label =5, then it convered to -1 -1 -1 -1 1 -1 -1 ..... until numer of tasks.
      for (int i=0;i<num;++i){
        for(int j=0; j<num_offset;++j){
          int d_idx =i*num_offset+j;
          int label = bottom_label[i]==j? 1:-1;
          if (label > 0) {
          // Update Positive accuracy and count
            true_positive += (bottom_data[d_idx] >= 0);
            false_negative += (bottom_data[d_idx] < 0);
            count_pos++;
          }
          if (label < 0) {
          // Update Negative accuracy and count
            true_negative += (bottom_data[d_idx] < 0);
            false_positive += (bottom_data[d_idx] >= 0);
            count_neg++;
          }
        }
      }
  }
 else{
    for (int ind = 0; ind < count; ++ind) {
      // Accuracy
      int label = static_cast<int>(bottom_label[ind]);
      if (label > 0) {
      // Update Positive accuracy and count
        true_positive += (bottom_data[ind] >= 0);
        false_negative += (bottom_data[ind] < 0);
        count_pos++;
      }
      if (label < 0) {
      // Update Negative accuracy and count
        true_negative += (bottom_data[ind] < 0);
        false_positive += (bottom_data[ind] >= 0);
        count_neg++;
      }
    }
}

  // Dtype AUC =0;

  // for (int ind = 0; ind < count; ++ind){
     // AUC=AUC + count_neg*true_positive + 0.5*DFP*DTP;
  // }
  Dtype sensitivity = (count_pos > 0)? (true_positive / count_pos) : 0;
  Dtype specificity = (count_neg > 0)? (true_negative / count_neg) : 0;
  Dtype harmmean = ((count_pos + count_neg) > 0)?
    2 / (count_pos / true_positive + count_neg / true_negative) : 0;
  Dtype precission = (true_positive > 0)?
    (true_positive / (true_positive + false_positive)) : 0;
  Dtype f1_score = (true_positive > 0)?
    2 * true_positive /
    (2 * true_positive + false_positive + false_negative) : 0;

  DLOG(INFO) << "Sensitivity: " << sensitivity;
  DLOG(INFO) << "Specificity: " << specificity;
  DLOG(INFO) << "Harmonic Mean of Sens and Spec: " << harmmean;
  DLOG(INFO) << "Precission: " << precission;
  DLOG(INFO) << "F1 Score: " << f1_score;
  (*top)[0]->mutable_cpu_data()[0] = sensitivity;
  (*top)[0]->mutable_cpu_data()[1] = specificity;
  (*top)[0]->mutable_cpu_data()[2] = harmmean;
  (*top)[0]->mutable_cpu_data()[3] = precission;
  (*top)[0]->mutable_cpu_data()[4] = f1_score;

  // MultiLabelAccuracy should not be used as a loss function.
  return Dtype(0);
}

INSTANTIATE_CLASS(MultiLabelAccuracyLayer);

}  // namespace caffe
