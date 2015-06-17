#include <cuda_runtime.h>

#include <string>

#include "caffe/caffe.hpp"
//#include "caffe/proto/caffe.pb.h"
//#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"


using namespace caffe;  // NOLINT(build/namespaces)
//using caffe::BlobProto;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc <3) {
    LOG(ERROR) << "Usage: " << string(argv[0]) <<"  height width vgg_mean.binaryproto" ;
    return 1;
  }
  printf("generating VGG model required data mean binary fil\n\n");
  int height 	 =atoi(argv[1]);
  int width  	 =atoi(argv[2]);
  int c_channels =30;
  int num        =1;
  
  
  float BGR_Mean[]       ={103.939, 116.779, 123.68};
  
  
  Blob<float> data_mean(num,c_channels,height,width,1);
  float* im_data = data_mean.mutable_cpu_data();
  int count =0;
  for (int c = 0; c < c_channels; ++c) {
          for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
			      im_data[count]=BGR_Mean[c%3];
				  count++;
            }
          }
        }
  
  
  caffe::BlobProto proto;
  data_mean.ToProto(&proto);
  WriteProtoToBinaryFile(proto, string(argv[3]));
  
  return 0;
}