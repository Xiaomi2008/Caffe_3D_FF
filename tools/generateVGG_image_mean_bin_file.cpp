// This code generate mean file for caffe net such as VGG mean file
//By Tao Zeng Dec 2014. @cs.odu
// Example:  generateVGG_image_mean_bin_file  <height> <width> <R> <G> <B> <outputfile>


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
    LOG(ERROR) << "Usage: " << string(argv[0]) <<"  <height> <width> <R> <G> <B> <outputfile>" ;
    return 1;
  }
  printf("generating VGG model required data at %s\n\n",argv[6]);
  int height 	 =atoi(argv[1]);
  int width  	 =atoi(argv[2]);
  int c_channels =3;
  int num        =1;
  
  float BGR_Mean[]={123,123,123};
 // float BGR_Mean[]       ={103.939, 116.779, 123.68};
 if (argc>3)
 {
   BGR_Mean[0]=atof(argv[5]);
   BGR_Mean[1]=atof(argv[4]);
   BGR_Mean[2]=atof(argv[3]);
	
 }else{
	//float BGR_Mean[]       ={123, 123, 123};
    BGR_Mean[0]=103.939;
    BGR_Mean[1]=116.779;
    BGR_Mean[2]=123.68;
  }
  
  Blob<float> data_mean(num,c_channels,height,width,1);
  float* im_data = data_mean.mutable_cpu_data();
  int count =0;
  for (int c = 0; c < c_channels; ++c) {
          for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
			      im_data[count]=BGR_Mean[c];
				  count++;
            }
          }
        }
  
  
  caffe::BlobProto proto;
  data_mean.ToProto(&proto);
  if (argc>3){
   WriteProtoToBinaryFile(proto, string(argv[6]));
   }
   else{
  WriteProtoToBinaryFile(proto, string(argv[3]));}
  
  return 0;
}