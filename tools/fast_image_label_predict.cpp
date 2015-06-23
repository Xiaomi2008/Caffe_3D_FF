#include <stdio.h>  // for snprintf
#include <cuda_runtime.h>
#include <google/protobuf/text_format.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <boost/algorithm/string.hpp>
#include <string>
#include <vector>
#include <float.h>
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
  //return 0;
    //return feature_extraction_pipeline<float>(argc, argv);
    return feature_extraction_pipeline<float>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  const int num_required_args = 11;
  if (argc < num_required_args) {
    LOG(ERROR)<<
    "This program takes in a trained network and an input data levelDB , and then"
    " elem-wise predict the output image produced by the net.\n"
    "Usage: patch_based_predict_3d  source_DB_name "
    "  deploy_proto_file  pretrained_model patch_h path_w path_d out_feature_name( usually last layer) out_put_db_name "
    "  [CPU/GPU] [DEVICE_ID=0][MEAN_FILE/MEAN_VALUE] [FILE_PATH/VALUE] \n";
    return 1;
  }
  int arg_pos = num_required_args;
  float mean  =.0;
  bool useMeanFile =false; 
  string mean_file;
  
  arg_pos = num_required_args;
  if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
    LOG(ERROR)<< "Using GPU";
    uint device_id = 0;
	
    // if (argc > arg_pos + 1) {
      // device_id = atoi(argv[arg_pos + 1]);
      // CHECK_GE(device_id, 0);
    // }
	 arg_pos++;
	 if (argc > arg_pos) {
        device_id = atoi(argv[arg_pos]);
        CHECK_GE(device_id, 0);
      }
	 arg_pos++;
    LOG(ERROR) << "Using Device_id=" << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
	
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
	 if(argc > arg_pos){mean = atof(argv[arg_pos + 1]);}
  }
  
  if(argc>arg_pos){
		if(strcmp(argv[arg_pos], "MEAN_FILE") == 0){
			
			arg_pos++;
			useMeanFile =true;
			mean_file=argv[arg_pos];
		}else if (strcmp(argv[arg_pos], "MEAN_VALUE") == 0){
		    arg_pos++;
            if(argc > arg_pos)			
			mean = atof(argv[arg_pos]);
		}
	}
  
  
  
  Caffe::set_phase(Caffe::TEST);

  arg_pos = 0;  // the name of the executable
  string source_DB_name(argv[++arg_pos]);
  string path_based_deploy_proto(argv[++arg_pos]);
  string pretrained_binary_proto(argv[++arg_pos]);
  int patch_h  =atoi(argv[++arg_pos]);
  int patch_w  =atoi(argv[++arg_pos]);
  int patch_d  =atoi(argv[++arg_pos]); 
  int start_h  = 0;
  LOG(INFO)<<patch_h<<" "<<patch_w<<" "<<patch_d;
  string predict_range(argv[++arg_pos]); 
	 
 // if(predict_range== "SINGLE"){
     start_h=atoi(argv[++arg_pos]);
  //}
  
  
  LOG(INFO)<<predict_range;
   
  LOG(INFO)<< start_h;
 // sleep(10);
  
  string predict_blob_name(argv[++arg_pos]);
  string out_put_db_name(argv[++arg_pos]);
  LOG(INFO)<<predict_blob_name;
  
  int soft_max_predict =atoi(argv[++arg_pos]);
  //sleep(10);
  //string predict_blob_name(argv[++arg_pos]);

 
  	// string feature_extraction_proto(argv[++arg_pos]);
  shared_ptr<Net<Dtype> > patch_prediction_net(
      new Net<Dtype>(path_based_deploy_proto));
  //feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);
  patch_prediction_net->CopyTrainedLayersWithMoreSourceChannelFrom(pretrained_binary_proto);
  //patch_prediction_net->CopyTrainedLayersFrom(pretrained_binary_proto);
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
	CHECK(iter);
	CHECK(iter->Valid());
    source_datum.ParseFromString(iter->value().ToString());
	//return 0;  
   
  
	
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
	//iter.reset(source_db->NewIterator(leveldb::ReadOptions()));
    //iter->SeekToFirst();
	
	const int kMaxKeyStrLength = 100;
    char key_str[kMaxKeyStrLength];
	string value;
	
	
	
	
	
	
	
	
	
	//========================================================
	
	
	
	
	
	
	
	
	
    //int num_mini_batches = 1024;
	LOG(INFO)<< "labeling images ...";
    //const int d_height =source_datum.height()-patch_h+1;
	//const int d_width  =source_datum.width()-patch_w+1;
	//const int d_depth  =source_datum.depth()-patch_d+1;
	//const int d_height =1;
	//const int d_width  =patch_w;
	//const int d_depth  =patch_d;
	//const int d_channels  =source_datum.channels();
	
	const int s_height =source_datum.height();
	const int s_width  =source_datum.width();
	const int s_depth  =source_datum.depth();
	const int s_channels  =source_datum.channels();
	int p_start_h   = start_h;
    int p_start_w   = 0;
	int p_start_d   = 0;
	int p_end_h   = patch_h;
    int p_end_w   = patch_w;
	int p_end_d   = patch_d;
	
	if(s_height < patch_h)
	{
		p_start_h=(s_height-patch_h)/2;
		p_end_h  =p_start_h +patch_h;
	}
	
	
	if(predict_range== "FULL"){
		p_start_h= 0-(patch_h/2);
		p_end_h  =s_height-(patch_h/2)+1;
		}
	else if(predict_range=="SINGLE"){
	   //p_end_h= p_start_h+1;
	   p_end_h=p_start_h+1;
	}

	
	
	
	
	if(s_width < patch_w)
	{
	  p_start_w=(s_width-patch_w)/2;
		p_end_w  =p_start_w +patch_w;
	}
	
	if(s_depth < patch_d)
	{
	  p_start_d=(s_depth-patch_d)/2;
	  p_end_d  =p_start_d +patch_d;
	}
	
	
	
	BlobProto blob_proto;
	Blob<Dtype> data_mean;
	Dtype* mean_data;
	if(useMeanFile){
			ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
			data_mean.FromProto(blob_proto);
			CHECK_EQ(data_mean.num(), 1);
			CHECK_EQ(data_mean.channels(), s_channels);
			CHECK_EQ(data_mean.height(), s_height);
			CHECK_EQ(data_mean.width(), s_width);
			CHECK_EQ(data_mean.depth(), s_depth);
			mean_data = data_mean.mutable_cpu_data();
		}
	else
		{
	        data_mean.Reshape(1,s_channels,s_height,s_width,s_depth);
		    mean_data = data_mean.mutable_cpu_data();
		    caffe_set(s_channels*s_height*s_width*s_depth, mean, mean_data);
		}
	
	
	
	const shared_ptr<Blob<Dtype> > predict_blob_temp 
				= patch_prediction_net->blob_by_name(predict_blob_name);
				
	
					//Dtype* pd;
				//int dim =predict_blob->count()/predict_blob->num();
				//LOG(INFO)<<"predict blob dim is "<<dim;
				//int dim =predict_blob->chanels();
	
	
	
	const int d_height =predict_blob_temp->height();
	const int d_width  =predict_blob_temp->width();
	const int d_depth  =predict_blob_temp->depth();
	const int d_channels  =source_datum.channels();
	
	if(predict_range=="FULL"){
		predict_datum.set_height(s_height);
		}
	else if(predict_range=="SINGLE"){
	  // predict_datum.set_height(d_height);
	   predict_datum.set_height(p_end_h-p_start_h-1);
	}
    
	
    predict_datum.set_width(d_width);
    predict_datum.set_depth(d_depth);
    predict_datum.set_channels(d_channels);
	//predict_datum.set_num(source_datum.channels());
	
	
	
	const string& source_data = source_datum.data();
	//int*  predict_data     = predict_datum.float_data();
	//shared_ptr<Blob<Dtype> > input_blob;
	shared_ptr<Blob<Dtype> > patch_blob;
	 //patch_blob.reset(new Blob<Dtype>(1 , source_datum.channels(), patch_h, patch_w, patch_d));
    // Dtype* patch_data = patch_blob->mutable_cpu_data();
	//patch_blob.reset(new Blob<Dtype>(1, source_datum.channels(), patch_h, patch_w, patch_d));
	
	
	
   
     LOG(INFO)<<"channel x height x width x depth = " << d_channels<<" "<<d_height <<" "<< d_width<<" "<<d_depth;
	 LOG(INFO)<<"channel x pacth_h x pacth_w x pacth_d = " << d_channels<<" "<<patch_h <<" "<< patch_w<<" "<<patch_d;
	 //sleep(2);
	 //input_blob.reset(new Blob<Dtype>(1 , source_datum.channels(), s_height, s_width, s_depth));
	 //Dtype* patch_data = input_blob->mutable_cpu_data();
	 int skip =0;
	 if(!skip){
	 //for (int h=0; h<1; h+=patch_h){
	// LOG(INFO)<<"processing
	  for (int h=p_start_h; h< p_end_h; h++){
	       LOG(INFO)<<"start h "<<h;
		   LOG(INFO)<<"processing height = " << h;
	   //for (int h=0; h<d_height; h+=patch_h){
		for (int w=0; w<d_width; w+=patch_w){
			
			//for (int d=0; d<1; d+=patch_d)
			for (int d=0; d<d_depth; d+=patch_d){
			  size_t count =0;
			   patch_blob.reset(new Blob<Dtype>(1 , source_datum.channels(), patch_h, patch_w, patch_d));
			   Dtype* patch_data = patch_blob->mutable_cpu_data();
			   //LOG(INFO)<<"processing depth = " << d;
				for (int c_p=0; c_p<d_channels; c_p++){
					for (int h_p=0; h_p<patch_h; h_p++){
					//for (int h_p=0; h_p<1; h_p++){
						for (int w_p=p_start_w; w_p<p_end_w; w_p++){
							for (int d_p=p_start_d; d_p<p_end_d; d_p++){
							   //int p_idx = (( c_p * patch_h + h_p)* patch_w + w_p) * patch_d+d_p;
							  // int s_idx =((c * s_height + h + h_p) * s_width + w + w_p)*s_depth+d+d_p+n;
							  size_t s_idx = (((size_t)c_p * (size_t)s_height + (size_t)h + (size_t)h_p) * (size_t)s_width + (size_t)w + (size_t)w_p)*(size_t)s_depth+(size_t)d+(size_t)d_p;
							   
							   
							  // patch_data[count]=source_data[s_idx]-mean_data[s_idx];
							   //int s_idx =((c * s_height + h + h_p) * s_width + w + w_p)*s_depth+d+d_p+n;
							   Dtype datum_element;
							   if(source_data.size()){
							        if(h+h_p>=0 && h+h_p<s_height && w+w_p>=0 && w+w_p<s_width&& d+d_p>=0 && d+d_p<s_depth )
									    //CHECK_LE(s_dix,)
										{datum_element=static_cast<Dtype>(static_cast<uint8_t>(source_data[s_idx]));
										patch_data[count]=datum_element-mean_data[s_idx];}
									else
										{datum_element = 0;
										patch_data[count]=0;}
								}else
								{
								    
									if(h+h_p>=0 && h+h_p<s_height && w+w_p>=0 && w+w_p<s_width&& d+d_p>=0 && d+d_p<s_depth )
										{datum_element=static_cast<Dtype>(static_cast<float_t>(source_data[s_idx]));
										patch_data[count]=datum_element-mean_data[s_idx];}
									else{
									  datum_element = 0;
									  patch_data[count]=0;
									  }
								}
							  
							   //CHECK_EQ(patch_data[count],source_data[s_idx]);
							   count++;
							   //patch_data[2]=2;
							}
						}
					}
				}
				//continue;
			    //LOG(INFO)<<"propagate height x width X d = " << h <<" "<<w<<" "<< d;				
	 
				vector<Blob<Dtype>*> input_vec;
				input_vec.clear();
				input_vec.push_back(patch_blob.get());
				patch_prediction_net->Forward(input_vec);
				LOG(INFO)<<"end forawrding";
				const shared_ptr<Blob<Dtype> > predict_blob 
				= patch_prediction_net->blob_by_name(predict_blob_name);
				
				const Dtype* pd=predict_blob->cpu_data();
					//Dtype* pd;
				//int dim =predict_blob->count()/predict_blob->num();
				//LOG(INFO)<<"predict blob dim is "<<dim;
				//int dim =predict_blob->chanels();
				int num =predict_blob->num();
				int chs =predict_blob->channels();
				int wid =predict_blob->width();
				int dep =predict_blob->depth();
				int ht  =predict_blob->height();
				
				//LOG(INFO)<<" p_h ="<<p_start_h<<" "<<p_end_h;
				//LOG(INFO)<<" p_w ="<<p_start_w<<" "<<p_end_w;
				//LOG(INFO)<<" p_d ="<<p_start_d<<" "<<p_end_d;
				
				
				LOG(INFO)<<"predict blob dim is "<<num<<" "<<chs<<" "<<ht<<" "<<" "<<wid<<" "<<dep;
				//soft_max_predict
				//bool soft_max_predict = false;
				if(soft_max_predict==0)
				{
				    predict_datum.set_channels(predict_blob->channels());
					for(int i=0; i<predict_blob->count();++i)
					   predict_datum.add_float_data(pd[i]);
				}
				else
				{
					for(int n=0;n<num;n++){
						
						
						for(int h=0; h<ht; h++){
							for(int w=0; w<wid; w++){
								for(int d=0; d<dep; d++){
									int max_idx=0; 
									Dtype max_v= - FLT_MAX;
									for(int c=0;c<chs;c++){
												size_t idx = ((((size_t)c*(size_t)ht +(size_t)h)*(size_t)wid+(size_t)w)*(size_t)dep+(size_t)d);
												if (max_v<pd[idx]) {max_v=pd[idx]; max_idx=c;}
											
										}
										
									//if(max_idx !=0) {LOG(INFO)<< "class is "<< max_idx;}
									predict_datum.add_float_data(max_idx);
								}
						}
						
					}
					//LOG(INFO)<<"added to datum";
					
					 //LOG(INFO)<<"predict label is "<<max_idx;
					  //CHECK_GE(predict_blob->count(),1);
					 //predict_data->push_back(max_idx);

						}
				}
			 
		}
	 
   }
   
  }
     //predict_datum.SerializeToString(&value);
     //LOG(INFO)<<"try to serialize predict_datum";
	 predict_datum.SerializeToString(&value);
	 LOG(INFO)<<"done serialize predict_datum";
	 snprintf(key_str, kMaxKeyStrLength, "%9d", 0);
	 feature_batches[0]->Put(string(key_str), value);
	 output_db->Write(write_options,feature_batches[0].get());
     LOG(INFO)<< "Successfully extracted the features!";
 }
  
  return 0;
}