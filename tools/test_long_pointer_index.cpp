#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include <iostream>

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
using namespace caffe;  // NOLINT(build/namespaces)

//template<typename Dtype>
//int feature_extraction_pipeline(int argc, char** argv);


int main(int argc, char** argv) {
   ::google::InitGoogleLogging(argv[0]);
  //return 0;
    //return feature_extraction_pipeline<double>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
    typedef long long unsigned int LONG;
    int k=3;
    int cube_k=k*k*k;
    size_t ch=100;
    size_t h=1;
    size_t w=1024;
    size_t d=1024;
	float xx =-FLT_MAX;
    LOG(INFO)<<"Min is "<<xx;
    Blob<float> mem_b;
   // int result_int = h*w*d*ch;
   // long long unsigned int result_long = (long long unsigned int)(h*w*d*ch);
   // //size_t result_t = (size_t)h*(size_t)w*(size_t)d*(size_t)ch;
    // size_t result_t =h*w*d*ch;
   // LOG(INFO)<<result_int;
   // LOG(INFO)<<result_t;
   // LOG(INFO)<<result_long;
    mem_b.Reshape(1,ch,h,w,d);
    float* x=mem_b.mutable_cpu_data();
	for (size_t i =0;i<mem_b.count();i++)
	{
		if (x[i]!=0){
		LOG(INFO)<<"x ["<<i<<"] = "<<x;
			sleep(1);
		}
	}

   //int* x =new int[result_t];
   //delete x;
   //if( (x =malloc(result))==NULL);
   //{LOG(INFO)<<"alloc failed ..." ;}
   //CHECK_GT(x,0);
   //free(x);//elete x;
  // std::cout << "result =" << result << std::endl;
   
}
