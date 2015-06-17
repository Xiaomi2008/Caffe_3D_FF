// Copyright 2014 BVLC and contributors.
//
// This is a simple script that allows one to quickly finetune a network.
// Usage:
//    finetune_net solver_proto_file pretrained_net

#include <cuda_runtime.h>

#include <string>

#include "caffe/caffe.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc <1) {
    LOG(ERROR) << "Usage: sectionInfoTxtfile";
    return 1;
  }
  printf("start tune \n\n");
  //int deviceID =0;
  //if (argc==4){
  //    deviceID =atoi(argv[3]);
  //}
  //Caffe::SetDevice(deviceID);
  ISH_section_param ISHsectionIMinfo;
  ReadProtoFromTextFileOrDie(argv[1], &ISHsectionIMinfo);
  //ReadProtoFromTextFile(argv[1], &ISHsectionIMinfo);
  
  //ISH_section_info
   //param.layers_size()
   //LOG(INFO)<<ISHsectionIMinfo
   
   int isize =ISHsectionIMinfo.ish_section_info_size();
   LOG(INFO)<<"there are "<< isize<<" of sections:";
  for (int section_idx = 0; section_idx < ISHsectionIMinfo.ish_section_info_size(); ++section_idx) {
    
	const ISH_section& section_param = ISHsectionIMinfo.ish_section_info(section_idx);
	 int secid =section_param.sectionid();
	 LOG(INFO)<<"section ID = "<<secid;
	for(int im_idx=0;im_idx<section_param.image_size();++im_idx){
		const imageInfo& im_param = section_param.image(im_idx);
		    LOG(INFO)<<"image path = "<<im_param.filename();
			}
	LOG(INFO)<<"label = " << section_param.labels();
	
	
	std::istringstream iss(section_param.labels());
	int label;
	while (iss >> label) {
     // labels.push_back(label);
	  LOG(INFO)<<label;
    }
 }
  //NetParameter param;
  //LOG(INFO)<<"InsertSplits(in_param, &param)";
  
  //InsertSplits(in_param, &param);
  // printf("error here \n\n");
  //LOG(INFO) << "Starting Optimization";
  //SGDSolver<float> solver(solver_param);
  //LOG(INFO) << "Loading from " << argv[2];
  //solver.net()->CopyTrainedLayersFrom(string(argv[2]));
  //solver.Solve();
  //LOG(INFO) << "Optimization Done.";

  return 0;
}