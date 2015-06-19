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
void SoftmaxWithLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
  if (top->size() >= 1) {
    // softmax loss (averaged across batch)
    (*top)[0]->Reshape(1, 1, 1, 1);
  }
  if (top->size() == 2) {
    // softmax output
    (*top)[1]->Reshape(bottom[0]->num(), bottom[0]->channels(),
        bottom[0]->height(), bottom[0]->width(), bottom[0]->depth());
  }
}

template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  //LOG(INFO)<<"SoftmaxLOSSLayer forward_pgu ";
  softmax_bottom_vec_[0] = bottom[0];
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  
  size_t outer_num = bottom[0]->num();
  size_t inner_num = bottom[0]->height()*bottom[0]->width()*bottom[0]->depth();
  size_t channels  = bottom[0]->channels();
  size_t dims=prob_.count() / outer_num;
  size_t num = prob_.num();
  size_t dim = prob_.count() / num;
  //LOG(INFO)<<" n "<<prob_.num()<<" c "<<prob_.channels()<<" h "<<prob_.height() <<" w "<<prob_.width() <<" d "<<prob_.depth() <<" outer " <<outer_num << " inner " <<inner_num;
  Dtype loss = 0;
  //LOG(INFO)<< "softmax loss layer prob num ="<<num<< " Dim ="<<dim<<  "  count ="<<prob_.count() <<"  FLT_MIN ="<<FLT_MIN //<<"  FLT_MAX ="<< FLT_MAX;
  
  size_t count = 0;
 if (inner_num>1){
	  for (size_t i = 0; i < outer_num; ++i) {
		for (size_t j = 0; j < inner_num; j++) {
		  const int label_value = static_cast<int>(label[i * inner_num + j]);
		  //if (has_ignore_label_ && label_value == ignore_label_) {
		  //  continue;
		  //}
		 // LOG(INFO)<<"label in loss is " <<label_value;
		  CHECK_GE(label_value, 0);
		  CHECK_LT(label_value, prob_.channels());
		  loss -= log(std::max(prob_data[i * dims + label_value * inner_num + j],
							   Dtype(FLT_MIN)));
		  ++count;
		}
	  }
  
  
  }
  else{
	  for (size_t i = 0; i < num; ++i) {
	   
		const int label_value=static_cast<int>(label[i]); 
		// current data's multi class is represent from 0-n , where 0 means such label
		//is not exist, and 1-n represents the #class n-1, Therefore, to comptibale with softmax which count class from 0:n-1
		// the input label value is minused by 1.
		//if (m_label>=0){  //for multilabel case, the code is modified by Tao Zeng 12/17/2014. we only count loss on the label    that is greater than /equal to zero . 
		      //if (label[i]!=0)
		//	LOG(INFO)<<"label in loss is " <<label[i];
			
		   CHECK_GE(label_value, 0);
		   CHECK_LT(label_value, prob_.channels());
			loss += -log(max(prob_data[i * dim + label_value],
							 Dtype(FLT_MIN)));
			//			 }
			
			//loss += -log(max(prob_data[i * dim + m_label],
			//				 Dtype(FLT_MIN)));
			//			 }
		//}
	  }
	  //sleep(2);
  
  }
  loss=loss/num;//(outer_num*inner_num);
  //LOG(INFO)<<"label in loss is " <<(*top)[0];
  if (top->size() >= 1) {
    //(*top)[0]->mutable_cpu_data()[0] = loss / num;
	
	//(*top)[0]->mutable_cpu_data()[0] = loss / num;
	(*top)[0]->mutable_cpu_data()[0] = loss;
	
  }
  if (top->size() == 2) {
    (*top)[1]->ShareData(prob_);
  }
  //LOG(INFO)<<"Done SoftmaxLOSSLayer forward_pgu ";
  //return loss / num;
   return loss;
}

typedef std::pair<int, float> PAIR;
struct CmpByValue {
  bool operator()(const PAIR& lhs, const PAIR& rhs)
  {return lhs.second > rhs.second;}
 };

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  //LOG(INFO)<<"SoftmaxLOSSLayer backward_pgu ";
  size_t outer_num = (*bottom)[0]->num();
  size_t inner_num = (*bottom)[0]->height()*(*bottom)[0]->width()*(*bottom)[0]->depth();
  size_t dims=prob_.count() / outer_num;  

   CHECK_GT(inner_num,0);
   CHECK_EQ((*bottom)[1]->count(),outer_num*inner_num);

  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  
  
  
  if (propagate_down[0]) {
		Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
		const Dtype* prob_data = prob_.cpu_data();
		const Dtype* label = (*bottom)[1]->cpu_data();
    if(inner_num <2)
	{     //LOG(INFO)<<"loss inner_num ="<<inner_num;
		//Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
		//const Dtype* prob_data = prob_.cpu_data();
		caffe_copy(prob_.count(), prob_data, bottom_diff);
		
		int num = prob_.num();
		int dim = prob_.count() / num;
		for (int i = 0; i < num; ++i) {
		  bottom_diff[i * dim + static_cast<int>(label[i])] -= 1;
		}
		// Scale down gradient
		caffe_scal(prob_.count(), Dtype(1) / num, bottom_diff);
	}
	else{

		//LOG(INFO)<<"loss inner_num ="<<inner_num;
		caffe_copy(prob_.count(), prob_data, bottom_diff);
		//const Dtype* label = (*bottom)[1]->cpu_data();
		size_t dim = prob_.count() / outer_num;
		size_t count = 0;
		
		float heigh_v =0;
		int heigh_freq_lb=0;
		for (size_t i = 0; i < outer_num; ++i) {
		   std::map<int,float> label_freq;
		   std::map<int,float>::iterator it;
		   for (size_t j = 0; j < inner_num; ++j){
		      //label_freq[label_value]=0;
			  const int label_value = static_cast<int>(label[i * inner_num + j]);
			  label_freq[label_value]=0;
			  //label_freq[label_value]++;
		   }
		   
		   for (size_t j = 0; j < inner_num; ++j){
				const int label_value = static_cast<int>(label[i * inner_num + j]);
			  label_freq[label_value]++;
		   }
		   
		   //vector<PAIR> label_freq_vec(label_freq.begin(), label_freq.end());  
			//sort(label_freq_vec.begin(), label_freq_vec.end(), CmpByValue());
		   //int low_lb= label_freq[label_freq.size()-1].first;
		   //float low_freq =label_freq[label_freq.size()-1].second;
		   //find highest freq label in patch; 
		  
		 
			   
		  for (size_t j = 0; j < inner_num; ++j) {
			const int label_value = static_cast<int>(label[i * inner_num + j]);
			//if (has_ignore_label_ && label_value == ignore_label_) {
			 // for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
			 //   bottom_diff[i * dim + c * inner_num_ + j] = 0;
			 // }
		   // } else {
		       Dtype lb_diff =bottom_diff[i * dims + label_value * inner_num + j]-1;
		       bottom_diff[i * dims + label_value * inner_num + j] =lb_diff;
			   for(it = label_freq.begin();it !=label_freq.end();it++){
		          int    lb				=it->first;
				  if(lb==label_value) continue;
				  float freq_prob		=it->second/inner_num;
				  Dtype prob_diff 		= bottom_diff[i * dims + lb * inner_num + j]-1;
				  bottom_diff[i * dims + lb * inner_num + j] =prob_diff*freq_prob*(0-lb_diff);
		  	   }
			  
		      // for(int k=0;k<label_freq.size();++k)
			  // {
			     // int lb= label_freq[k].first;
				
				 // bottom_diff[i * dims + lb * inner_num + j] =prob_diff *
			  // }
			  
		      
		      // Dtype dif_m =bottom_diff[i * dims + heigh_freq_lb * inner_num + j] -1;
			  // Dtype dif_lb =bottom_diff[i * dims + label_value * inner_num + j] -1;
			  // bottom_diff[i * dims + heigh_freq_lb * inner_num + j]=dif_m*dif_lb;
			  // bottom_diff[i * dims + label_value * inner_num + j] =dif_lb;
			  
		      //bottom_diff[i * dims + heigh_freq_lb * inner_num + j] -= 1;
			  //bottom_diff[i * dims + label_value * inner_num + j] -= 1;
			  ++count;
			}
		  }
		
		
		// // Scale down gradient
		//caffe_scal(prob_.count(), Dtype(1) / count, bottom_diff);
		caffe_scal(prob_.count(), Dtype(1) / count, bottom_diff);
	}
  
}
//LOG(INFO)<<"Done SoftmaxLOSSLayer backward_pgu ";
}
INSTANTIATE_CLASS(SoftmaxWithLossLayer);

}  // namespace caffe
