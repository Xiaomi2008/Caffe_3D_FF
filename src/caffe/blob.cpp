// Copyright 2014 BVLC and contributors.

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

#include <stdio.h>
namespace caffe {

template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height,
    const int width, const int depth) {
  CHECK_GE(num, 0);
  CHECK_GE(channels, 0);
  CHECK_GE(height, 0);
  CHECK_GE(width, 0);
  CHECK_GE(depth, 0);
  bool reset =true;
  if(num_ == num&&  height_ == height &&width_ == width && depth_==depth){
     reset =false;
  }else{
	  num_ = num;
	  channels_ = channels;
	  height_ = height;
	  width_ = width;
	  depth_ = depth;
	  reset =true;
  } 
  count_ = (size_t)num_* (size_t)channels_ * (size_t)height_ * (size_t)width_ * (size_t)depth_;
  //int all_count =count_ * sizeof(Dtype);
  //LOG(INFO)<<"mem count is "<<count_ ;
  //printf(" mem count is %d\n", (int)count_);
  if(count_==0)
  {
   data_.reset(reinterpret_cast<SyncedMemory*>(NULL));
   diff_.reset(reinterpret_cast<SyncedMemory*>(NULL));
  }else if(reset||data_==NULL){
	data_.reset(new SyncedMemory(count_ * sizeof(Dtype)));
    diff_.reset(new SyncedMemory(count_ * sizeof(Dtype)));
  }
  
  CHECK(data_);
  //else{
  
  //}
  // if (count_) {
    // data_.reset(new SyncedMemory(count_ * sizeof(Dtype)));
    // diff_.reset(new SyncedMemory(count_ * sizeof(Dtype)));
  // } else {
    // data_.reset(reinterpret_cast<SyncedMemory*>(NULL));
    // diff_.reset(reinterpret_cast<SyncedMemory*>(NULL));
  // }
}

template <typename Dtype>
void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {
  Reshape(other.num(), other.channels(), other.height(), other.width(), other.depth());
}

template <typename Dtype>
Blob<Dtype>::Blob(const int num, const int channels, const int height,
    const int width, const int depth) {
  Reshape(num, channels, height, width, depth);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->cpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data) {
  CHECK(data);
  data_->set_cpu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
  CHECK(data_);
  //LOG(INFO)<<"get mutable_cpu_data()";
  return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template <>
size_t* Blob<size_t>::mutable_cpu_data() {
  CHECK(data_);
  return static_cast<size_t*>(data_->mutable_cpu_data());
}


template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_gpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_cpu_data());
}



template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_gpu_data());
}

template <typename Dtype>
void Blob<Dtype>::ShareData(const Blob& other) {
  CHECK_EQ(count_, other.count());
  data_ = other.data();
}

template <typename Dtype>
void Blob<Dtype>::ShareDiff(const Blob& other) {
  CHECK_EQ(count_, other.count());
  diff_ = other.diff();
}

// The "update" method is used for parameter blobs in a Net, which are stored
// as Blob<float> or Blob<double> -- hence we do not define it for
// Blob<int> or Blob<unsigned int>.
template <> void Blob<unsigned int>::Update() { NOT_IMPLEMENTED; }
template <> void Blob<int>::Update() { NOT_IMPLEMENTED; }

template <typename Dtype>
void Blob<Dtype>::Update() {
  // We will perform update based on where the data is located.
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    caffe_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->cpu_data()),
        static_cast<Dtype*>(data_->mutable_cpu_data()));
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
    // perform computation on GPU
    caffe_gpu_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->gpu_data()),
        static_cast<Dtype*>(data_->mutable_gpu_data()));
    break;
  default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
}

template <typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
  if (num_ != source.num() || channels_ != source.channels() ||
      height_ != source.height() || width_ != source.width()||depth_!=source.depth()) {
    if (reshape) {
      Reshape(source.num(), source.channels(), source.height(), source.width(), source.depth());
    } else {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
    }
  }
  switch (Caffe::mode()) {
  case Caffe::GPU:
    if (copy_diff) {
	//LOG(INFO)<<"diff_->mutable_gpu_diff()";
      caffe_copy(count_, source.gpu_diff(),
          static_cast<Dtype*>(diff_->mutable_gpu_data()));
    } else {
	 // LOG(INFO)<<"data_->mutable_gpu_data()";
      caffe_copy(count_, source.gpu_data(),
          static_cast<Dtype*>(data_->mutable_gpu_data()));
    }
    break;
  case Caffe::CPU:
    if (copy_diff) {
      caffe_copy(count_, source.cpu_diff(),
          static_cast<Dtype*>(diff_->mutable_cpu_data()));
    } else {
      caffe_copy(count_, source.cpu_data(),
          static_cast<Dtype*>(data_->mutable_cpu_data()));
    }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto) {
  LOG(INFO)<<"FromProto size = "<<proto.num()  <<" "<<proto.channels() << " "<<proto.height()<<" "<< proto.width()<<" "<< proto.depth();
  if(proto.depth() ==0)
  {
		LOG(INFO)<< "proto depth is 0, converting to 1 for 2D models to 3D models...";
		Reshape(proto.num(), proto.channels(), proto.height(), proto.width(), 1);
  }else{
		Reshape(proto.num(), proto.channels(), proto.height(), proto.width(), proto.depth());
  }
  // copy data
  Dtype* data_vec = mutable_cpu_data();
  /* for testing only
  Dtype data_vec_sum=0;
  for (int i = 0; i < count_; ++i) {
    data_vec_sum=data_vec_sum+data_vec[i]; 
  }
  LOG(INFO)<<"bolb sum value = "<<data_vec_sum;
  */
  for (size_t i = 0; i < count_; ++i) {
    data_vec[i] = proto.data(i);
	//LOG(INFO)<<"proto.data(i) ="<<proto.data(i);
  }
  //LOG(INFO)<<"proto.data[11870779] ="<<proto.data(11870779);
  //sleep(20);
  //  LOG(INFO)<<"proto.diff_size= "<<proto.diff_size();
  if (proto.diff_size() > 0) {
    Dtype* diff_vec = mutable_cpu_diff();
    for (size_t i = 0; i < count_; ++i) {
      diff_vec[i] = proto.diff(i);
    }
  }
  
  //if (Caffe::mode()==Caffe::GPU) {
  // gpu_data();
  // }
  
   
}

template <typename Dtype>
void Blob<Dtype>::FromProtoWithExtraCopy(const BlobProto& proto){
 // reshape other dimention but remain the same number of channels;
  Reshape(proto.num(), this->channels(), proto.height(), proto.width(), proto.depth());
  // copy data
  Dtype* data_vec = mutable_cpu_data();
  int sourceDataSize =proto.data_size();
  for (int i = 0; i < count_; ++i) {
    data_vec[i] = proto.data(i%sourceDataSize);
  }
  if (proto.diff_size() > 0) {
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.diff(i%sourceDataSize);
    }
  }
}


template <typename Dtype>
void Blob<Dtype>::FromProtoFC2Conv(const BlobProto& proto){
 // Note that we do not change the shape  of taget layer blob (convolution)
 // reshape other dimention but remain the same number of channels;
  //Reshape(proto.num(), this->channels(), proto.height(), proto.width(), proto.depth());
  // copy data
  Dtype* data_vec = mutable_cpu_data(); 
  size_t proto_count =proto.num()*proto.channels()*proto.height()*proto.width()*proto.depth();
  CHECK_EQ(proto_count,count_);
  for (size_t i = 0; i < count_; ++i) {
    data_vec[i] = proto.data(i);
  }
  //  LOG(INFO)<<"proto.diff_size= "<<proto.diff_size();
  if (proto.diff_size() > 0) {
    Dtype* diff_vec = mutable_cpu_diff();
    for (size_t i = 0; i < count_; ++i) {
      diff_vec[i] = proto.diff(i);
    }
  }
}



template <typename Dtype>
void Blob<Dtype>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->set_num(num_);
  proto->set_channels(channels_);
  proto->set_height(height_);
  proto->set_width(width_);
  proto->set_depth(depth_);
  proto->clear_data();
  proto->clear_diff();
  const Dtype* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_data(data_vec[i]);
  }
  if (write_diff) {
    const Dtype* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_diff(diff_vec[i]);
    }
  }
}

//template <typename Dtype>
// void Blob<Dtype>::release_data(){
  // data_->release();
// }

template <typename Dtype>
void Blob<Dtype>::release_gpu_data(){
  data_->release_gpu_data();
}

template <typename Dtype>
void Blob<Dtype>::release_all_data(){
  data_->release_all_data();
}

INSTANTIATE_CLASS(Blob);
template class Blob<int>;
template class Blob<unsigned int>;

}  // namespace caffe

