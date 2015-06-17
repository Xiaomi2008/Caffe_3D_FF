// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
class Blob {
 public:
  Blob()
       : data_(), diff_(), num_(0), channels_(0), height_(0), width_(0),depth_(0),
       count_(0) {}
  explicit Blob(const int num, const int channels, const int height,
    const int width,const int depth);
  void Reshape(const int num, const int channels, const int height,
    const int width, const int depth = 1);
  void ReshapeLike(const Blob& other);
  inline size_t num() const { return num_; }
  inline size_t channels() const { return channels_; }
  inline size_t height() const { return height_; }
  inline size_t width() const { return width_; }
  inline size_t depth() const{return depth_;}
  inline size_t count() const {return count_; }
  //inline int offset(const int n, const int c = 0, const int h = 0,
  //    const int w = 0, const int d = 0) const {
  inline size_t offset(const int n, const int c = 0, const int h = 0,
      const int w = 0, const int d = 0) const {
    CHECK_GE(n, 0);
    CHECK_LE(n, num_);
    CHECK_GE(channels_, 0);
    CHECK_LE(c, channels_);
    CHECK_GE(height_, 0);
    CHECK_LE(h, height_);
    CHECK_GE(width_, 0);
    CHECK_LE(w, width_);
	CHECK_GE(depth_, 0);
    CHECK_LE(d, depth_);
	//return (size_t)((((n * channels_ + c) * height_ + h) * width_ + w) * depth_ + d);
   // return ((n * channels_ + c) * height_ + h) * width_ + w;
   return (size_t)(((((size_t)n * (size_t)channels_ + (size_t)c) * (size_t)height_ + (size_t)h) * (size_t)width_ + (size_t)w) * (size_t)depth_ + (size_t)d);
  }
  // Copy from source. If copy_diff is false, we copy the data; if copy_diff
  // is true, we copy the diff.
  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);

  inline Dtype data_at(const int n, const int c, const int h,
      const int w,const int d) const {
    return *(cpu_data() + offset(n, c, h, w, d));
  }

  inline Dtype diff_at(const int n, const int c, const int h,
      const int w, const int d) const {
    return *(cpu_diff() + offset(n, c, h, w, d));
  }

  inline const shared_ptr<SyncedMemory>& data() const {
    CHECK(data_);
    return data_;
  }

  inline const shared_ptr<SyncedMemory>& diff() const {
    CHECK(diff_);
    return diff_;
  }
  void release_gpu_data();
  //void release_cpu_data();
  void release_all_data();
  const Dtype* cpu_data() const;
  void set_cpu_data(Dtype* data);
  const Dtype* gpu_data() const;
  const Dtype* cpu_diff() const;
  const Dtype* gpu_diff() const;
  Dtype* mutable_cpu_data();
  Dtype* mutable_gpu_data();
  Dtype* mutable_cpu_diff();
  Dtype* mutable_gpu_diff();
  void Update();
  void FromProto(const BlobProto& proto);
  
  // Tao Zeng added: 12/2/2014:
  // this functions were created to deal with case that we want multple images (eg. slices of brains) to be stored as multiple channels
  // and read the pretrained model which has only one image(3 channels-RGB )
  // this function will just copy rest of weighs correspoding to one images RBG channel to all other channels
  void FromProtoWithExtraCopy(const BlobProto& proto);
  void ToProto(BlobProto* proto, bool write_diff = false) const;
  
  
  // Tao Zeng added: 22/5/2015:
  // this functions are mode specifically for copy full connection parameters to convolution parameters.
  // This usually happend whe try to do elem_wise predict, where the train model have full connection layer while
  // the testing/prediction models use convolution layer to do element_wise prediciton
  void FromProtoFC2Conv(const BlobProto& proto);

  // Set the data_/diff_ shared_ptr to point to the SyncedMemory holding the
  // data_/diff_ of Blob other -- useful in layers which simply perform a copy
  // in their forward or backward pass.
  // This deallocates the SyncedMemory holding this blob's data/diff, as
  // shared_ptr calls its destructor when reset with the = operator.
  void ShareData(const Blob& other);
  void ShareDiff(const Blob& other);

 protected:
  shared_ptr<SyncedMemory> data_;
  shared_ptr<SyncedMemory> diff_;
  size_t num_;
  size_t channels_;
  size_t height_;
  size_t width_;
  size_t depth_;
  size_t count_;
 

  DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
