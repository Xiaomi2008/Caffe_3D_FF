#include <stdio.h>  // for snprintf
#include <cuda_runtime.h>
#include <google/protobuf/text_format.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <boost/algorithm/string.hpp>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);


int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  const int num_required_args = 9;
  if (argc < num_required_args) {
    LOG(ERROR)<<
    "This program takes in a trained network and an input data levelDB , and then"
    " elem-wise predict the output image produced by the net.\n"
    "Usage: patch_based_predict_3d  source_DB_name "
    "  deploy_proto_file  pretrained_model patch_h path_w path_d out_feature_name( usually last layer) out_put_db_name"
    "  [CPU/GPU] [DEVICE_ID=0]\n";
    return 1;
  }
  int arg_pos = num_required_args;

  arg_pos = num_required_args;
  if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
    LOG(ERROR)<< "Using GPU";
    uint device_id = 0;
    if (argc > arg_pos + 1) {
      device_id = atoi(argv[arg_pos + 1]);
      CHECK_GE(device_id, 0);
    }
    LOG(ERROR) << "Using Device_id=" << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }
  Caffe::set_phase(Caffe::TEST);

  arg_pos = 0;  // the name of the executable
  string source_DB_name(argv[++arg_pos]);
  string path_based_deploy_proto(argv[++arg_pos]);
  string pretrained_binary_proto(argv[++arg_pos]);
  int patch_h  =atoi(argv[++arg_pos]);
  int patch_w  =atoi(argv[++arg_pos]);
  int patch_d  =atoi(argv[++arg_pos]);
  string predict_blob_name(argv[++arg_pos]);
  string out_put_db_name(argv[++arg_pos]);
  //string predict_blob_name(argv[++arg_pos]);

 // string feature_extraction_proto(argv[++arg_pos]);
  shared_ptr<Net<Dtype> > patch_prediction_net(
      new Net<Dtype>(path_based_deploy_proto));
  //feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);
 // patch_prediction_net->CopyTrainedLayersWithMoreSourceChannelFrom(pretrained_binary_proto);
  patch_prediction_net->CopyTrainedLayersFrom(pretrained_binary_proto);
  CHECK(patch_prediction_net->has_blob(predict_blob_name))
        << "Unknown feature blob name " << predict_blob_name
        << " in the network " << path_based_deploy_proto;

  
  
// Read first source image(3d) from source level_DB;
  Datum source_datum;
  Datum predict_datum;
  leveldb::Options options; 
  shared_ptr<leveldb::Iterator> iter;
  options.create_if_missing = false;
  options.max_open_files = 100;
  //options.write_buffer_size = 268435456;
 
  shared_ptr<leveldb::DB> source_db;
    LOG(INFO)<< "Opening source leveldb " << source_DB_name;
    leveldb::DB* db;
    leveldb::Status status = leveldb::DB::Open(options,
                                               source_DB_name.c_str(),
                                               &db);
    CHECK(status.ok()) << "Failed to open leveldb " << source_DB_name;
	source_db.reset(db);
	iter.reset(source_db->NewIterator(leveldb::ReadOptions()));
    iter->SeekToFirst();
	source_datum.ParseFromString(iter->value().ToString());
	
	
	
	
	////////////////////////////////////////////////-------------------------
  int num_features =1;
    vector<shared_ptr<leveldb::WriteBatch> > feature_batches(num_features);
  for (int i = 0; i < num_features; ++i) {
    feature_batches[i] = shared_ptr<leveldb::WriteBatch>(
      new leveldb::WriteBatch());
  }
	
  leveldb::DB* db_w;
  leveldb::WriteOptions write_options;
  write_options.sync = true;
  options.error_if_exists = false;
  options.create_if_missing = true;
  //write_options.create_if_missing = true;
  shared_ptr<leveldb::DB> output_db;
  status = leveldb::DB::Open(options, out_put_db_name.c_str(),&db_w);
    CHECK(status.ok()) << "Failed to open leveldb " << out_put_db_name;
	output_db.reset(db_w);
	iter.reset(source_db->NewIterator(leveldb::ReadOptions()));
    iter->SeekToFirst();
	
	const int kMaxKeyStrLength = 100;
    char key_str[kMaxKeyStrLength];
	string value;
	
	
	
	
	
	
	
	
	
	//========================================================
	
	
	
	
	
	
	
	
	
    int num_mini_batches = 1024;
	LOG(INFO)<< "labeling images ...";
    const int d_height =source_datum.height()-patch_h+1;
	const int d_width  =source_datum.width()-patch_w+1;
	const int d_depth  =source_datum.depth()-patch_d+1;
	const int d_channels  =source_datum.channels();
	
	const int s_height =source_datum.height();
	const int s_width  =source_datum.width();
	const int s_depth  =source_datum.depth();
	const int s_channels  =source_datum.channels();
	
	predict_datum.set_height(d_height);
    predict_datum.set_width(d_width);
    predict_datum.set_depth(d_depth);
    predict_datum.set_channels(source_datum.channels());
	//predict_datum.set_num(source_datum.channels());
	
	
	
	const string& source_data = source_datum.data();
	//int*  predict_data     = predict_datum.float_data();
	shared_ptr<Blob<Dtype> > patch_blob;
	//patch_blob.reset(new Blob<Dtype>(1, source_datum.channels(), patch_h, patch_w, patch_d));
	
	
	
   
    LOG(INFO)<<"channel x height x width x depth = " << d_channels<<" "<<d_height <<" "<< d_width<<" "<<d_depth;
	 LOG(INFO)<<"channel x pacth_h x pacth_w x pacth_d = " << d_channels<<" "<<patch_h <<" "<< patch_w<<" "<<patch_d;

//for (int c=0; c<d_channels; ++c){
	for (int h=25; h<d_height; ++h){
	    
		for (int w=0; w<d_width; ++w){
			LOG(INFO)<<"processing height x width = " << h <<" "<<w;
			//for (int d=0; d<d_depth; d=d+num_mini_batches){
			int count =0;
			patch_blob.reset(new Blob<Dtype>(num_mini_batches , source_datum.channels(), patch_h, patch_w, patch_d));
			Dtype* patch_data = patch_blob->mutable_cpu_data();
			for (int d=0; d<d_depth; ++d){
			    
				//int top_index = (( c* d_height + h)* d_width + w) *d_depth +d;
			    //LOG(INFO)<<"start get patch ....";
		   
		   //int 
			//for(int n=0;n<num_mini_batches;n++){
				for (int c_p=0; c_p<d_channels; c_p++){
					for (int h_p=0; h_p<patch_h; h_p++){
						for (int w_p=0; w_p<patch_w; w_p++){
							for (int d_p=0; d_p<patch_d; d_p++){
							   //int p_idx = (( c_p * patch_h + h_p)* patch_w + w_p) * patch_d+d_p;
							  // int s_idx =((c * s_height + h + h_p) * s_width + w + w_p)*s_depth+d+d_p+n;
							  int s_idx = ((c_p * s_height + h + h_p) * s_width + w + w_p)*s_depth+d+d_p;
							  
							  //int s_idx =((c * s_height + h + h_p) * s_width + w + w_p)*s_depth+d+d_p+n;
							   patch_data[count]=source_data[s_idx];
							   count++;
							   //LOG(INFO)<<"see serial idx =" <<p_idx;
							}
						}
					}
				}
				
			}
			
			//LOG(INFO)<<"end get patch ....";
				vector<Blob<Dtype>*> input_vec;
				//LOG(INFO)<<"start forwarding ....";
                input_vec.push_back(patch_blob.get());
				
				//LOG(INFO)<<"finish get patch data  at " << h <<" " <<w <<" "<<d;
				patch_prediction_net->Forward(input_vec);
				//LOG(INFO)<<"end forawrding";
				const shared_ptr<Blob<Dtype> > predict_blob 
				= patch_prediction_net->blob_by_name(predict_blob_name);
				
				  Dtype* pd=predict_blob->mutable_cpu_data();
					//Dtype* pd;
				int dim =predict_blob->count()/predict_blob->num();
			    //LOG(INFO)<<"predict blob dim is "<<dim;
				//int dim =predict_blob->chanels();
				int num =predict_blob->num();
				
				//int org =((c * s_height + patch_h) * s_width + w + patch_w)*s_depth+d+patch_d+n;
				//int org =((c * s_height + h+patch_h/2) * s_width + w + patch_w/2)*s_depth+d+patch_d/2;
				//int org =((c * s_depth + d+patch_d/2) * s_width + w + patch_w/2)*s_height+h+patch_h/2;
			   // int s_data =source_data[org];
					 //if (s_data<0){LOG(INFO)<<"s_data is " << s_data;} 
				//predict_datum.add_float_data(s_data);
				
				for(int n=0;n<num;n++){
				    int max_idx=0; 
					Dtype max_v= -99999999;
					for(int j=0; j<dim; j++)
					{
						 if (max_v<pd[j+(n*dim)]) {max_v=pd[j+(n*dim)]; max_idx=j;}
					 }
					
					
					
					
					 //LOG(INFO)<<"predict label is "<<max_idx;
					   CHECK_GE(predict_blob->count(),1);
					// predict_data->push_back(max_idx);
					 predict_datum.add_float_data(max_idx);
					 
					 //int org =((c * s_height + patch_h) * s_width + w + patch_w)*s_depth+d+patch_d+n;
					 //int s_data =source_data[org];
					 //if (s_data<0){LOG(INFO)<<"s_data is " << s_data;} 
					  //predict_datum.add_float_data(s_data);
				 }
				// LOG(INFO)<<"end predict";
				//datum.add_float_data(feature_blob_data[d]);
				//predict_datuma.dd_float_data(pd[0]);
				//patch_data
			
			
			
		}
		
		 predict_datum.SerializeToString(&value);
		snprintf(key_str, kMaxKeyStrLength, "%9d", 0);
		
		//LOG(INFO)<<"Key is "<<key_str;
		feature_batches[0]->Put(string(key_str), value);
	//predict_datum.ParseFromString(iter->value().ToString());
	
		output_db->Write(write_options,
                                feature_batches[0].get());
		
		
	}
 //}

  
   // shared_ptr<leveldb::WriteBatch> predict_batch;
   // =shared_ptr<leveldb::WriteBatch>(new leveldb::WriteBatch());
	  
	
  // const int kMaxKeyStrLength = 100;
  // char key_str[kMaxKeyStrLength];
  // vector<Blob<float>*> input_vec;
  // vector<int> image_indices(num_features, 0);
  
  // leveldb::WriteOptions write_options;
  // write_options.sync = true;
  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    // feature_extraction_net->Forward(input_vec);
	// LOG(INFO)<<"extracting # "<<batch_index <<"features" << "out of "<< num_mini_batches;
    // for (int i = 0; i < num_features; ++i) {
      // const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net
          // ->blob_by_name(blob_names[i]);
      // int batch_size = feature_blob->num();
      // int dim_features = feature_blob->count() / batch_size;
	  // //LOG(INFO)<<"feature "<< blob_names[i] << "size is "<< dim_features;
      // Dtype* feature_blob_data;
      // for (int n = 0; n < batch_size; ++n) {
       // // datum.set_height(dim_features);
		// datum.set_height(feature_blob->height());
        // datum.set_width(feature_blob->width());
		// datum.set_depth(feature_blob->depth());
        // datum.set_channels(feature_blob->channels());
        // datum.clear_data();
        // datum.clear_float_data();
        // feature_blob_data = feature_blob->mutable_cpu_data() +
            // feature_blob->offset(n);
        // for (int d = 0; d < dim_features; ++d) {
          // datum.add_float_data(feature_blob_data[d]);
        // }
        // string value;
        // datum.SerializeToString(&value);
        // snprintf(key_str, kMaxKeyStrLength, "%9d", image_indices[i]);
		
		// //LOG(INFO)<<"Key is "<<key_str;
        // feature_batches[i]->Put(string(key_str), value);
        // ++image_indices[i];
		
        // if (image_indices[i] % 1000 == 0) {
         // feature_dbs[i]->Write(leveldb::WriteOptions(),
                                // feature_batches[i].get());
		 // // feature_dbs[i]->Write(write_options,
         // //                       feature_batches[i].get());
          // LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
              // " query images for feature blob " << blob_names[i];
          // feature_batches[i].reset(new leveldb::WriteBatch());
        // }
      // }  // for (int n = 0; n < batch_size; ++n)
    // }  // for (int i = 0; i < num_features; ++i)
  // }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
  // // write the last batch
  // for (int i = 0; i < num_features; ++i) {
    // if (image_indices[i] % 1000 != 0) {
      // //feature_dbs[i]->Write(leveldb::WriteOptions(), feature_batches[i].get());
	  // feature_dbs[i]->Write(write_options,
                                // feature_batches[i].get());
    // }
    // LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
        // " query images for feature blob " << blob_names[i];
  // }

  LOG(ERROR)<< "Successfully extracted the features!";
  return 0;
}