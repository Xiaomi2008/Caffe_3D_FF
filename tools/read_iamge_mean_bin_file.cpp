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
    LOG(ERROR) << "Usage: " << string(argv[0]) <<" vgg_mean.binaryproto " <<"Channels" ;
    return 1;
  }
  printf("read  mean binary fil\n\n");
  
  Blob<float> data_mean;
  
  
  
  caffe::BlobProto proto;
  
  ReadProtoFromBinaryFile(string(argv[1]),&proto);
  data_mean.FromProto(proto);
  float* im_data = data_mean.mutable_cpu_data();
  
  int height     =data_mean.height();
  int width      =data_mean.width();
  int c_channels =data_mean.channels();
  int num        =data_mean.num();
  int count      =data_mean.count();
  int count_point =0;
  int c= atoi(argv[2]);
  //for (int c = atoi(argv[2])-1; c < atoi(argv[2]); ++c) {
  int baseIdx =c*width*height;
          for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
			      //im_data[count]=BGR_Mean[c];
				 int idx =baseIdx+count_point;
				 LOG(INFO) <<im_data[idx];
				  count_point++;
            }
          }
    //    } 
	 LOG(INFO)<<"Height =" << height <<" width =" <<width <<" channels =" <<c_channels <<" num =" << num << " count =" << count;
  return 0;
}