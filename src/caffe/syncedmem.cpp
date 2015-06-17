// Copyright 2014 BVLC and contributors.

#include <cuda_runtime.h>

#include <cstring>

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_);
  }

  if (gpu_ptr_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
}

void SyncedMemory::release_gpu_data(){
  // if (cpu_ptr_ && own_cpu_data_) {
  //  CaffeFreeHost(cpu_ptr_);
 // }
  to_cpu();
  if (gpu_ptr_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
  head_ =HEAD_AT_CPU;
  gpu_ptr_=NULL;
  
}

void SyncedMemory::release_all_data(){
  //to_cpu();
  if (gpu_ptr_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
	//LOG(INFO)<<"gpu data released ...";
  }
  gpu_ptr_=NULL;
 // head_ =HEAD_AT_CPU;
  
 
  if (cpu_ptr_) {
		//if (own_cpu_data_) {
		//LOG(INFO)<<"cpu data released ...";
		CaffeFreeHost(cpu_ptr_);
        cpu_ptr_=NULL; 
		
		//}
  }
  head_ =UNINITIALIZED;
}

// void SyncedMemory::release_all_data(){
  // //to_cpu();
  // if (cpu_ptr_) {
		// if (own_cpu_data_) {
		// LOG(INFO)<<"cpu data released ...";
		// CaffeFreeHost(cpu_ptr_);
		// head_ =HEAD_AT_CPU;
        // cpu_ptr_=NULL;
	  // }
  // }
// }





inline void SyncedMemory::to_cpu() {
  switch (head_) {
  case UNINITIALIZED:
    //LOG(INFO)<<"mem is in UNINITIALIZED " <<"size will be "<<size_;
    CaffeMallocHost(&cpu_ptr_, size_);
    memset(cpu_ptr_, 0, size_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_);
      own_cpu_data_ = true;
    }
    caffe_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() {
  switch (head_) {
  case UNINITIALIZED:
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    CUDA_CHECK(cudaMemset(gpu_ptr_, 0, size_));
    head_ = HEAD_AT_GPU;
	//LOG(INFO)<<"INI DATA";
    break;
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    }
	//LOG(INFO)<<"to_gpu()";
    caffe_memcpy(size_, cpu_ptr_, gpu_ptr_);
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
}

const void* SyncedMemory::cpu_data() {
  to_cpu();
  return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {
  CHECK(data);
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data() {
  to_gpu();
  return (const void*)gpu_ptr_;
}

void* SyncedMemory::mutable_cpu_data() {
  // LOG(INFO)<<"calling mutable_cpu_data";
  // switch (head_) {
  // case UNINITIALIZED:
	// LOG(INFO)<<"MEM not initilaized";
	// break;
  // case HEAD_AT_CPU:
     // LOG(INFO)<<"MEM is in CPU";
	// break;
  // case HEAD_AT_GPU:
    // LOG(INFO)<<"MEM is in GPU";
	// break;
	// }
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
  // LOG(INFO)<<"calling mutable_gpu_data";
  // switch (head_) {
  // case UNINITIALIZED:
	// LOG(INFO)<<"MEM not initilaized";
	// break;
  // case HEAD_AT_CPU:
     // LOG(INFO)<<"MEM is in CPU";
	// break;
  // case HEAD_AT_GPU:
    // LOG(INFO)<<"MEM is in GPU";
	// break;
	// }
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
}


}  // namespace caffe

