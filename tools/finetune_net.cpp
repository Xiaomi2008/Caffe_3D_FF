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
  if (argc <3) {
    LOG(ERROR) << "Usage: finetune_net solver_proto_file pretrained_net deviceID";
    return 1;
  }
  printf("start tune \n\n");
  int deviceID =0;
  if (argc==4){
      deviceID =atoi(argv[3]);
  }
  Caffe::SetDevice(deviceID);
  SolverParameter solver_param;
  ReadProtoFromTextFileOrDie(argv[1], &solver_param);
  // printf("error here \n\n");
  LOG(INFO) << "Starting Optimization";
  SGDSolver<float> solver(solver_param);
  LOG(INFO) << "Loading from " << argv[2];
  
  // Original function --------------------------------
  //solver.net()->CopyTrainedLayersFrom(string(argv[2]));
  //----------------------------------------------------
  
 //====================================================================================================================================
  // By Tao Zeng added: 12/2/2014:
  // this functions were created to deal with case that we want multple images (eg. slices of brains) to be stored as multiple channels
  // and read the pretrained model which has only one image(3 channels-RGB )
  // this function will just copy rest of weighs correspoding to one images RBG channel to all other channels
  //LOG(INFO)<<"load model "<<string(argv[2]);
 solver.net()->CopyTrainedLayersWithMoreSourceChannelFrom(string(argv[2]));
 //sleep(100);
 //======================================================================================================================================
  
  
  solver.Solve();
  LOG(INFO) << "Optimization Done.";

  return 0;
}
