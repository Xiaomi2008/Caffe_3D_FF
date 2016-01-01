// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>
//#include <mutex>
#include <string>
#include <vector>
#include <cmath>
#include <math.h>
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"

using std::string;

namespace caffe {
//pthread_mutex_t c_mutex;
template <typename Dtype>
void* ImageLabelDataLayerPrefetch(void* layer_pointer) {
  //LOG(INFO)<<"start run prefectch thread ";
  //printf("start run prefectch thread \n");

  CHECK(layer_pointer);
  ImageLabelDataLayer<Dtype>* layer = static_cast<ImageLabelDataLayer<Dtype>*>(layer_pointer);
  CHECK(layer);

  //mutex mtx;

 // layer_batch_info_struct<Dtype> *layer_batch_info
 // = static_cast<layer_batch_info_struct<Dtype> *>(layer_pointer);
 // CHECK(layer_batch_info);
 // ImageLabelDataLayer<Dtype>* layer =static_cast<ImageLabelDataLayer<Dtype>*>(layer_batch_info->layer_pointer);



  //LOG(INFO)<<" item start : item end =" <<item_start <<" "<<item_end;
  //Datum datum;


  //c_mute

  CHECK(layer->prefetch_data_);
 // pthread_mutex_lock(&c_mutex);
	Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
  //pthread_mutex_unlock(&c_mutex);
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
 // int num_labels = 0;
  if (layer->output_labels_) {
   // pthread_mutex_lock(&c_mutex);
    top_label = layer->prefetch_label_->mutable_cpu_data();
	//pthread_mutex_unlock(&c_mutex);
  }


  const Dtype scale = layer->layer_param_.image_label_data_param().scale();
  const int batch_size = layer->layer_param_.image_label_data_param().batch_size();
  //const int crop_size = layer->layer_param_.image_label_data_param().crop_size();
  const bool mirror = layer->layer_param_.image_label_data_param().mirror();
  const bool padding = layer->layer_param_.image_label_data_param().padding();


  //LOG(INFO)<<"prefetch_label_->count() =" <<layer->prefetch_label_->count();
  //sleep()
  //if (mirror && crop_size == 0)

  // if (mirror && layer->is_crop_)  {
  //  LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
  //      << "set at the same time.";
  //}
  // datum scales
  const int channels = layer->datum_channels_;
  const int height = layer->datum_height_;
  const int width = layer->datum_width_;
  const int depth = layer->datum_depth_;
  const int size = layer->datum_size_;
  int rand_patch_times =layer->rand_patch_times_;

  //const int channels = layer->datum_channels_;
  const int out_label_size_h = layer->out_label_size_h_;
  const int out_label_size_w = layer->out_label_size_w_;
  const int out_label_size_d = layer->out_label_size_d_;

  const int crop_size_h = layer->crop_size_h_;
  const int crop_size_w = layer->crop_size_w_;
  const int crop_size_d = layer->crop_size_d_;
  const bool is_crop    =layer->is_crop_;
  //const bool binary = layer->layer_param_.image_label_data_param().binary();
  //const int size = layer->datum_size_;
  //LOG(INFO)<<"current element label  size is: "<< out_label_size_h;
  //pthread_mutex_lock(&c_mutex);
  const Dtype* mean = layer->data_mean_.cpu_data();
  //LOG(INFO)<<"in thread mean[0] = "<< mean[0];
  //sleep(100);
 // pthread_mutex_unlock(&c_mutex);
  //if (rand_patch_times>batch_size){rand_patch_times=batch_size;}

  //unsigned int item_start = layer_batch_info->batch_start;
  //unsigned int item_end   = layer_batch_info->batch_end;
  unsigned int item_start = 0;
  unsigned int item_end   =  batch_size;
  size_t  label_size =layer->prefetch_label_->depth()*layer->prefetch_label_->width()*layer->prefetch_label_->height();
  //if(item_end >batch_size)
  //   item_end =batch_size;


  //for (int item_id = 0; item_id < batch_size; ++item_id) {
  // LOG(INFO)<<" 101 line";
  for (int item_id = item_start; item_id < item_end; ++item_id) {

	if( layer->read_patch_times_%rand_patch_times==0){
	     LOG(INFO)<<"reading new datum from LevelDB ...." <<"rand_patch_times =" <<rand_patch_times ;
		 //pthread_mutex_lock(&c_mutex);
	     layer->readNextDatums();
		// pthread_mutex_unlock(&c_mutex);
		 layer->read_patch_times_=0;
	}
	int datum_idx = layer->read_patch_times_ % layer->num_memory_datums_;
   // pthread_mutex_lock(&c_mutex);
	layer->read_patch_times_++;
   //pthread_mutex_unlock(&c_mutex);
   //LOG(INFO)<<"layer->datums_"<< datum_idx<< " "; ;
    Datum& datum=layer->datums_[datum_idx];
    const string& data = datum.data();
	//const string& elm_labels = datum.float_label();
	int label_h_off, label_w_off, label_d_off;
	label_h_off=label_w_off=label_d_off=0;
	//LOG(INFO)<<"layer->datums_"<< datum_idx<< " "; ;
    //if (crop_size) {
	if (is_crop) {
      CHECK(data.size()) << "Image cropping only support uint8 data";
      int h_off, w_off, d_off;
	  bool accept_label 	=false;
	  bool do_mirror    	=false;
    bool no_mirror      =true;
	  int mirro_dimention   =1; // 0=  h mirro, 1= w mirro, 2= d mirro
	  do{
	  int prefech_h = abs(layer->PrefetchRand());
	  int prefech_w = abs(layer->PrefetchRand());
	  int prefech_d = abs(layer->PrefetchRand());
		if(padding)
		{
		    if(layer->output_labels_){
				h_off = prefech_h  % (height-out_label_size_h/2)  -  (crop_size_h - out_label_size_h)/2;
				w_off = prefech_w  % (width -out_label_size_w/2)  -  (crop_size_w - out_label_size_w)/2 ;
				d_off = prefech_d  % (depth -out_label_size_d/2)  -  (crop_size_d - out_label_size_d)/2 ;

			}else{
				h_off = prefech_h  % height - crop_size_h/2;
				w_off = prefech_w  % width  - crop_size_w/2 ;
				d_off = prefech_d  % depth  - crop_size_d/2 ;
			}

		}else{
			h_off = prefech_h  % (height - crop_size_h);// +(height - crop_size)/4;
			w_off = prefech_w  % (width  - crop_size_w) ;//+(width - crop_size)/4;
			d_off = prefech_d  % (depth  - crop_size_d) ;//+ (width - crop_size)/4;
		}


		label_h_off = h_off + (crop_size_h-out_label_size_h)/2;
		label_w_off = w_off + (crop_size_w-out_label_size_w)/2;
		label_d_off = d_off + (crop_size_d-out_label_size_d)/2;

	//do_mirror=(mirror && layer->PrefetchRand() % 2);
  int pre_rand= abs(layer->PrefetchRand())%4;
  do_mirror=mirror &&(pre_rand<3);
	mirro_dimention=layer->PrefetchRand() % 3;
  //no_mirror  =
	if (layer->output_labels_){
      //if (mirror && layer->PrefetchRand() % 2)
	  if(do_mirror){
        // Copy mirrored version
		        size_t top_index=0;
				for (size_t c = 0; c < 1; ++c) {
				  for (size_t h = 0; h < out_label_size_h; ++h) {
					for (size_t w = 0; w < out_label_size_w; ++w) {
						for (size_t d = 0; d < out_label_size_d; ++d){
						  switch(mirro_dimention){
						  case 0:
						  top_index = (((item_id * 1 + c) * out_label_size_h + h)
										  * out_label_size_w + (out_label_size_w - 1 - w)) *out_label_size_d +d;
										break;
						  case 1:
						  top_index = (((item_id * 1 + c) * out_label_size_h +
												(out_label_size_h - 1 - h))
												* out_label_size_w + w) *out_label_size_d +d;
										break;
					      case 2:
						  top_index = (((item_id * 1 + c) * out_label_size_h +
												h)* out_label_size_w + w) *out_label_size_d
												+(out_label_size_d - 1 - d);
									    break;
						  default:
						    LOG(ERROR)<<"wrong mirro param";
						  }

						 size_t data_index = ((c * height + h + label_h_off) * width + w + label_w_off)*depth +d+label_d_off;
							  Dtype datum_element = static_cast<Dtype>(static_cast<float_t>(datum.float_label(data_index)));
							if(layer->balancing_label_)
							{
							  accept_label=layer->accept_given_label(datum_element);
							  if(accept_label ||layer->output_single_label_)
									top_label[top_index] = layer->get_converted_label(datum_element);

							}
							else
							 {
							     accept_label          = true;
							     top_label[top_index]  =datum_element;
							 }
						}
					}
				  }
				}
		}else{
		       // Not mirrorring....
		       // top_label[item_id]=10;
		    // if (layer->output_labels_){
				for (size_t c = 0; c < 1; ++c) {
				  for (size_t h = 0; h < out_label_size_h; ++h) {
					for (size_t w = 0; w < out_label_size_w; ++w) {
						for (size_t d = 0; d < out_label_size_d; ++d){
						 // int top_index = ((item_id * channels + c) * crop_size + h)
						 //				  * crop_size + w;
						  size_t top_index = (((item_id * 1 + c) * out_label_size_h + h)
										  * out_label_size_w + w) *out_label_size_d +d;
						  size_t data_index = ((c * height + h + label_h_off) * width + w + label_w_off)*depth +d+label_d_off;
							  Dtype datum_element = static_cast<Dtype>(static_cast<float_t>(datum.float_label(data_index)));
							if(layer->balancing_label_)
							{
							  accept_label=layer->accept_given_label(datum_element);
							  if(accept_label)
									top_label[top_index] = layer->get_converted_label(datum_element);
							  if(layer->output_single_label_)
								      top_label[top_index] = layer->get_converted_label(datum_element);
							}
							else
							 {
							     accept_label          = true;
							     top_label[top_index]  =datum_element;
							 }
						}
					}
				  }
				}
			}

		}

		if(label_size>1){break;}
	  }while(layer->output_labels_&&!accept_label);


	if(layer->output_single_label_){
	 do{
	    size_t l_h_off = label_h_off +out_label_size_h/2;
		size_t l_w_off = label_w_off +out_label_size_w/2;
		size_t l_d_off = label_d_off +out_label_size_d/2;
		size_t l_s_index = (l_h_off* width  + l_w_off)*depth+l_d_off;

		//size_t data_index = ((c * height + h + label_h_off) * width + w + label_w_off)*depth +d+label_d_off;
		Dtype* top_s_label =layer->prefetch_single_label_->mutable_cpu_data();
		Dtype label= static_cast<Dtype>(static_cast<float_t>(datum.float_label(l_s_index)));
		top_s_label[item_id] = label;//
			  if(layer->balancing_label_)
				{
				  accept_label=layer->accept_given_label(label);
				  if(accept_label)
						top_label[item_id] = layer->get_converted_label(label);
				}
				else
				 {
					 accept_label          = true;
					 top_label[item_id]  =label;
				 }
		}while( !accept_label);
	}


	//if (mirror && layer->PrefetchRand() % 2) {
	  if(do_mirror){
	     size_t top_index =0 ;
        for (size_t c = 0; c < channels; ++c) {
          for (size_t h = 0; h < crop_size_h; ++h) {
            for (size_t w = 0; w < crop_size_w; ++w) {
				for (size_t d = 0; d < crop_size_d; ++d){
				      //size_t top_index = (((item_id * channels + c) * crop_size_h + h)
					  //		  * crop_size_w + (crop_size_w - 1 - w))*crop_size_d +d;
					   switch(mirro_dimention){
						  case 0:
						  top_index = (((item_id * channels + c) * crop_size_h + h)
										  * crop_size_w + ( crop_size_w - 1 - w)) *crop_size_d +d;
										break;
						  case 1:
						  top_index = (((item_id * channels + c) * crop_size_h +
												(crop_size_h - 1 - h))
												* crop_size_w + w) *crop_size_d +d;
										break;
					      case 2:
						  top_index = (((item_id * channels + c) * crop_size_h +
												h)* crop_size_w  + w) *crop_size_d
												+(crop_size_d - 1 - d);
									    break;
						  default:
						    LOG(ERROR)<<"wrong mirro param";
						  }
				      int h_idx = h + h_off;
					  int w_idx = w + w_off;
					  int d_idx = d + d_off;
					  h_idx=h_idx<0? 0-h_idx:  h_idx;
					  w_idx=w_idx<0? 0-w_idx:  w_idx;
					  d_idx=d_idx<0? 0-d_idx:  d_idx;
					  h_idx=h_idx>=height? h_off+(crop_size_h-h-1) : h_idx;
					  w_idx=w_idx>=width? w_off+(crop_size_w-w-1): w_idx;
					  d_idx=d_idx>=depth? d_off+(crop_size_d-d-1) : d_idx;


					   size_t data_index = ((c * height + h_idx) * width + w_idx)*depth +d_idx;
					  //if(h_idx >= 0 && h_idx<height && w_idx >= 0 && w_idx < width && d_idx >= 0 && d_idx < depth){
							  Dtype datum_element;
							  if(data.size()){
									datum_element=static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));}
							  else{
									datum_element =datum.float_data(data_index);
								  }

							 top_data[top_index] = (datum_element - mean[data_index]) * scale;
							 //Dtype mean2= layer->data_mean_.cpu_data()[data_index];
							 //top_data[top_index] = (datum_element - mean2) * scale;
							// LOG(INFO)<<" Mirro crop |  top_data[top_index] = "<<top_data[top_index] <<"datum_element = "<<datum_element << "  mean["<<data_index<<"]= "<<mean[data_index]<<"  mean_bob "<< layer->data_mean_.cpu_data()[0];
						    // sleep(3);
						//}else{



							 // h_idx=h_idx<0? 0-h_idx:  h_off-h-1;
							 // w_idx=w_idx<0? 0-w_idx:  w_off-w-1;
							 // d_idx=d_idx<0? 0-d_idx:  d_off-d-1;
						    // data_index =((c * height + h_idx) * width + w_idx)*depth +d_idx;
							// top_data[top_index] = datum.float_data(data_index);
						//}
			  }
            }
          }
        }
      } else {
        // Normal copy
		// copy source image data
			for (size_t c = 0; c < channels; ++c) {
			  for (size_t h = 0; h < crop_size_h; ++h) {
				for (size_t w = 0; w < crop_size_w; ++w) {
					for (size_t d = 0; d < crop_size_d; ++d){
					 // int top_index = ((item_id * channels + c) * crop_size + h)
					 //				  * crop_size + w;
					  size_t top_index = (((item_id * channels + c) * crop_size_h + h)
									  * crop_size_w + w) *crop_size_d +d;

					  int h_idx = h + h_off;
					  int w_idx = w + w_off;
					  int d_idx = d + d_off;
					  h_idx=h_idx<0? 0-h_idx:  h_idx;
					  w_idx=w_idx<0? 0-w_idx:  w_idx;
					  d_idx=d_idx<0? 0-d_idx:  d_idx;
					  h_idx=h_idx>=height? h_off+(crop_size_h-h-1) : h_idx;
					  w_idx=w_idx>=width? w_off+(crop_size_w-w-1): w_idx;
					  d_idx=d_idx>=depth? d_off+(crop_size_d-d-1) : d_idx;
					  size_t data_index = ((c * height + h_idx) * width + w_idx)*depth +d_idx;
					  //if(h_idx >= 0 && h_idx<height && w_idx >= 0 && w_idx < width && d_idx >= 0 && d_idx < depth)
					  //{
							Dtype datum_element;
						   if(data.size()){
									datum_element=static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));}
						   else{
									datum_element =datum.float_data(data_index);
							}
						   //top_data[top_index] =  datum_element;
						     top_data[top_index] = (datum_element - mean[data_index]) * scale;
						     //Dtype mean2= layer->data_mean_.cpu_data()[data_index];
							 //top_data[top_index] = (datum_element - mean2) * scale;
							 // LOG(INFO)<<" Norm crop |  top_data[top_index] =  "<<top_data[top_index] <<"datum_element = "<<datum_element << "  mean["<<data_index<<"]= "<<mean[data_index]<<"  mean_bob "<< layer->data_mean_.cpu_data()[0];
						   //sleep(3);
						  // CHECK_GE(datum_element,0);
						  // CHECK_LE(datum_element,255);
						   //static_cast<Dtype>(static_cast<float_t>(data[data_index]));//datum_element* scale;//(datum_element - mean[data_index]) * scale;
					   //}else{
						// top_data[top_index] = 0;
						//}
					}
				}
			  }
			}
			// copy targe pixel(voxel) label data

      }
    } else {
      // we will prefer to use data() first, and then try float_data()
		  if (data.size()) {
			for (size_t j = 0; j < size; ++j) {
			  Dtype datum_element =
				  static_cast<Dtype>(static_cast<uint8_t>(data[j]));
			  top_data[item_id * size + j] = (datum_element - mean[j]) * scale;
			  //top_data[item_id * size + j] = (datum_element - 0) * scale;
			}
		  } else {
			for (size_t j = 0; j < size; ++j) {
			  top_data[item_id * size + j] =
				  (datum.float_data(j) - mean[j]) * scale;
				  //(datum.float_data(j) - 0) * scale;
			}
		  }

		  if (layer->output_labels_){
				label_h_off = 0+(height-out_label_size_h)/2;
				label_w_off = 0+(width-out_label_size_w)/2;
				label_d_off = 0+(depth-out_label_size_d)/2;
				 for (size_t c = 0; c < 1; ++c) {
				  for (size_t h = 0; h < out_label_size_h; ++h) {
					for (size_t w = 0; w < out_label_size_w; ++w) {
						for (size_t d = 0; d < out_label_size_d; ++d){

						  size_t top_index = (((item_id * 1 + c) * out_label_size_h + h)
										  * out_label_size_w + w) *out_label_size_d +d;

						  size_t data_index = ((c * height + h + label_h_off) * width + w + label_w_off)*depth +d+label_d_off;
						  Dtype datum_element =
							  static_cast<Dtype>(static_cast<float_t>(datum.float_label(data_index)));
						  top_label[top_index] = datum_element;
						}
					}
				  }
				}

		  }
    }

  }
 //LOG(INFO)<<"done prefectch thread ";
  //pthread_mutex_unlock(&c_mutex);
  return static_cast<void*>(NULL);
}

template <typename Dtype>
ImageLabelDataLayer<Dtype>::~ImageLabelDataLayer<Dtype>() {
  JoinPrefetchThread();
  // clean up the database resources
  switch (this->layer_param_.image_label_data_param().backend()) {
  case ImageLabelDataParameter_DB_LEVELDB:
    break;  // do nothing
  case ImageLabelDataParameter_DB_LMDB:
    mdb_cursor_close(mdb_cursor_);
    mdb_close(mdb_env_, mdb_dbi_);
    mdb_txn_abort(mdb_txn_);
    mdb_env_close(mdb_env_);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}

template <typename Dtype>
void ImageLabelDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  read_patch_times_ =0;
  if (top->size() >= 2) {
    output_labels_ = true;
  } else {
    output_labels_ = false;
  }
  if (top->size() == 3) {
    output_single_label_ = true;
  } else {
    output_single_label_ = false;
  }


  rand_patch_times_ =this->layer_param_.image_label_data_param().rand_patch_times();

  // Initialize DB
  switch (this->layer_param_.image_label_data_param().backend()) {
  case ImageLabelDataParameter_DB_LEVELDB:
    {
    leveldb::DB* db_temp;
    leveldb::Options options;
    options.create_if_missing = false;
    options.max_open_files = 100;
    LOG(INFO) << "Opening leveldb " << this->layer_param_.image_label_data_param().source();
    leveldb::Status status = leveldb::DB::Open(
        options, this->layer_param_.image_label_data_param().source(), &db_temp);
    CHECK(status.ok()) << "Failed to open leveldb "
                       << this->layer_param_.image_label_data_param().source() << std::endl
                       << status.ToString();
    db_.reset(db_temp);
	// iter_ is a memeber obj  =shared_ptr<leveldb::Iterator> iter_;
    iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
    iter_->SeekToFirst();
    }
    break;
  case ImageLabelDataParameter_DB_LMDB:
    CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env_,
             this->layer_param_.image_label_data_param().source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    LOG(INFO) << "Opening lmdb " << this->layer_param_.image_label_data_param().source();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_label_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.image_label_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      switch (this->layer_param_.image_label_data_param().backend()) {
      case ImageLabelDataParameter_DB_LEVELDB:
        iter_->Next();
        if (!iter_->Valid()) {
          iter_->SeekToFirst();
        }
        break;
      case ImageLabelDataParameter_DB_LMDB:
        if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
            != MDB_SUCCESS) {
          CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
                   MDB_FIRST), MDB_SUCCESS);
        }
        break;
      default:
        LOG(FATAL) << "Unknown database backend";
      }
    }
  }
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  switch (this->layer_param_.image_label_data_param().backend()) {
  case ImageLabelDataParameter_DB_LEVELDB:
    datum.ParseFromString(iter_->value().ToString());
    break;
  case ImageLabelDataParameter_DB_LMDB:
    datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  datum_channels_ 	= datum.channels();
  datum_height_ 	= datum.height();
  datum_width_ 		= datum.width();
  datum_depth_ 		= datum.depth();
  datum_size_ 		= datum.channels() * datum.height() * datum.width() * datum.depth();



  // image
  is_crop_ =false;
  if (this->layer_param_.image_label_data_param().has_crop_size()){
		crop_size_h_ = this->layer_param_.image_label_data_param().crop_size();
		if(crop_size_h_>0){
			crop_size_d_=crop_size_w_ = crop_size_h_;
			 // crop_size_h_;
			is_crop_ =true;
		}
	}else{
	   // if()
		crop_size_h_ =this->layer_param_.image_label_data_param().crop_size_h();
		crop_size_w_ =this->layer_param_.image_label_data_param().crop_size_w();
		crop_size_d_ =this->layer_param_.image_label_data_param().crop_size_d();
		if(crop_size_h_>0&& crop_size_w_>0 && crop_size_d_ >0){
		  is_crop_ =true;
		}
	}




  //int crop_size = this->layer_param_.image_label_data_param().crop_size();
  LOG(INFO) <<"Data layer input  batch_size is = "<<this->layer_param_.image_label_data_param().batch_size();
 // if (crop_size > 0)
   if (is_crop_){
	//CHECK_GT(datum_height_, crop_size_h_);
	//CHECK_GT(datum_width_, crop_size_w_);
	//CHECK_GT(datum_depth_, crop_size_d_);
    (*top)[0]->Reshape(this->layer_param_.image_label_data_param().batch_size(),
                       datum.channels(), crop_size_h_, crop_size_w_, crop_size_d_);
	// shared_ptr<Blob<Dtype> > prefetch_data_
    prefetch_data_.reset(new Blob<Dtype>(
        this->layer_param_.image_label_data_param().batch_size(), datum.channels(),
        crop_size_h_, crop_size_w_, crop_size_d_));
  } else {
		(*top)[0]->Reshape(
        this->layer_param_.image_label_data_param().batch_size(), datum.channels(),
        datum.height(), datum.width(), datum.depth());
    prefetch_data_.reset(new Blob<Dtype>(
        this->layer_param_.image_label_data_param().batch_size(), datum.channels(),
        datum.height(), datum.width(), datum.depth()));
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width()<< " ,"<<(*top)[0]->depth();
  // label
  if (output_labels_) {
    //CHECK_GT(datum.label_size(), 0) << "Datum should contain labels for top";
	if (this->layer_param_.image_label_data_param().has_out_label_size()){
		out_label_size_d_ =out_label_size_w_ = out_label_size_h_
		= this->layer_param_.image_label_data_param().out_label_size();
	}else{
		out_label_size_h_ =this->layer_param_.image_label_data_param().out_label_size_h();
		out_label_size_w_ =this->layer_param_.image_label_data_param().out_label_size_w();
		out_label_size_d_ =this->layer_param_.image_label_data_param().out_label_size_d();
	}
    (*top)[1]->Reshape(this->layer_param_.image_label_data_param().batch_size(),
      1, out_label_size_h_, out_label_size_w_, out_label_size_d_);
    LOG(INFO) << "output label size: " << (*top)[1]->num() << ","
      << (*top)[1]->channels() << "," << (*top)[1]->height() << ","
      << (*top)[1]->width()<< " ,"<<(*top)[1]->depth();
    //prefetch_label_.reset(
    //    new Blob<Dtype>(this->layer_param_.image_label_data_param().batch_size(),
    //      1,  datum.height(), datum.width(), datum.depth()));
     prefetch_label_.reset(
        new Blob<Dtype>(this->layer_param_.image_label_data_param().batch_size(),
          1,  out_label_size_h_, out_label_size_w_, out_label_size_d_));
	if(is_crop_)
	{
		CHECK_GE(crop_size_h_,out_label_size_h_);
		CHECK_GE(crop_size_w_,out_label_size_w_);
		CHECK_GE(crop_size_d_,out_label_size_d_);
	}
	    CHECK_GE(datum_height_,out_label_size_h_);
		CHECK_GE(datum_width_,out_label_size_w_);
		CHECK_GE(datum_depth_,out_label_size_d_);

  }

   if (output_single_label_) {
     (*top)[2]->Reshape(this->layer_param_.image_label_data_param().batch_size(),
      1, 1, 1, 1);
	  prefetch_single_label_.reset(new Blob<Dtype>(
        this->layer_param_.image_label_data_param().batch_size(), 1,
        1,1,1));

   }
  // datum size



  // check if we want to have mean

  if (this->layer_param_.image_label_data_param().has_mean_file()) {
    const string& mean_file = this->layer_param_.image_label_data_param().mean_file();
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.channels(), datum_channels_);
    CHECK_EQ(data_mean_.height(), datum_height_);
    CHECK_EQ(data_mean_.width(), datum_width_);
	CHECK_EQ(data_mean_.depth(), datum_depth_);
  } else {
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, datum_channels_, datum_height_, datum_width_,datum_depth_);
	data_mean_.cpu_data();
  }
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_->mutable_cpu_data();
  if (output_labels_)
    prefetch_label_->mutable_cpu_data();

  //LOG(INFO)<<"data_mean[0] ="<<data_mean_.cpu_data()[0];
  //sleep(20);
  num_memory_datums_= this->layer_param_.image_label_data_param().num_memory_datum();
  datums_.resize(num_memory_datums_);
  //readNextDatums();

  ProcessLabelSelectParam();
  LOG(INFO) << "Initializing prefetch";
  CreatePrefetchThread();
  LOG(INFO) << "Prefetch initialized.";
}

template<typename Dtype>
void ImageLabelDataLayer<Dtype>::readNextDatums(){
  switch (this->layer_param_.image_label_data_param().backend()) {
		case DataParameter_DB_LEVELDB:
			  for(int i=0;i<datums_.size();i++){

				  if (!iter_->Valid()) {
					// We have reached the end. Restart from the first.
					LOG(INFO) << "Restarting data prefetching from start.";
					iter_->SeekToFirst();
				  }
				// LOG(INFO)<<"data parsing ["<<i<<"]";
				  datums_[i].ParseFromString(iter_->value().ToString());
				 /// LOG(INFO)<<"data parsed";
				  iter_->Next();
				 // LOG(INFO)<<"iter->next";
			  }
		  break;
		case DataParameter_DB_LMDB:
		   for(int i=0;i<datums_.size();i++){
		   datums_[i].ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
			  if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
					  &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
				// We have reached the end. Restart from the first.
				DLOG(INFO) << "Restarting data prefetching from start.";
				CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
						&mdb_value_, MDB_FIRST), MDB_SUCCESS);
			  }

		   }
		  break;
		default:
		  LOG(FATAL) << "Unknown database backend";
	}
}



template<typename Dtype>
void ImageLabelDataLayer<Dtype>::ProcessLabelSelectParam(){
    //this->layer_param_
	num_labels_      =this->layer_param_.image_label_data_param().label_select_param().num_labels() ;
	balancing_label_ =this->layer_param_.image_label_data_param().label_select_param().balance();
	map2order_label_ =this->layer_param_.image_label_data_param().label_select_param().reorder_label();
	//num_top_label_balance_ =this->layer_param_.image_label_data_param().label_select_param().num_top_label_balance();
	//CHECK_EQ(is_label_blance,layer_param_.has_class_prob_mapping_file());
	//const string  label_prob_map_file = layer_param_.class_prob_mapping_file();
	bool has_prob_file=this->layer_param_.image_label_data_param().label_select_param().has_class_prob_mapping_file();
	if(balancing_label_)
	   {
		CHECK_EQ(balancing_label_,has_prob_file);
		const string&  label_prob_map_file = this->layer_param_.image_label_data_param().label_select_param().class_prob_mapping_file();
		ReadLabelProbMappingFile(label_prob_map_file);
		LOG(INFO)<<"Done ReadLabelProbMappingFile";
		compute_label_skip_rate();
		LOG(INFO)<<"compute_label_skip_rate()";
		}
}

template <typename Dtype>
void ImageLabelDataLayer<Dtype>::ReadLabelProbMappingFile(const string& source){
  Label_Prob_Mapping_Param lb_param;
  ReadProtoFromTextFileOrDie(source, &lb_param);
  ignore_rest_of_label_ 				= 	lb_param.ignore_rest_of_label();
  rest_of_label_mapping_ 				= 	lb_param.rest_of_label_mapping();
  rest_of_label_mapping_label_			=	lb_param.rest_of_label_mapping_label();
  rest_of_label_prob_					=	lb_param.rest_of_label_prob();
  num_labels_with_prob_  				= 	lb_param.label_prob_mapping_info_size();
  if(this->layer_param_.image_label_data_param().label_select_param().has_num_top_label_balance())
	num_top_label_balance_ =this->layer_param_.image_label_data_param().label_select_param().num_top_label_balance();
  else
	num_top_label_balance_ =  num_labels_with_prob_;

  CHECK_GE(num_top_label_balance_,1);
  CHECK_GE(num_labels_with_prob_,num_top_label_balance_);
  CHECK_GE(num_labels_,2);
  CHECK_GE(num_labels_,num_labels_with_prob_);
  LOG(INFO)<<"rest_of_label_mapping_  = "<<rest_of_label_mapping_<<" "<<rest_of_label_mapping_label_;
  label_prob_map_.clear();
  label_mapping_map_.clear();
  LOG(INFO)<< "label_prob_map_ size =" <<label_prob_map_.size();
  for (int i=0;i<num_labels_with_prob_;++i){
	const Label_Prob_Mapping&   label_prob_mapping_param = lb_param.label_prob_mapping_info(i);
	int   label 			=	label_prob_mapping_param.label();
	float lb_prob			=   label_prob_mapping_param.prob();
	int   mapped_label ;
	if(label_prob_mapping_param.has_map2label())
	    mapped_label =   label_prob_mapping_param.map2label();
	else
		 mapped_label = label ;
	label_prob_map_[label]	=	lb_prob;
	label_mapping_map_[label]=   mapped_label;

	//LOG(INFO)<<"label_mapping_map_["<<label<<"] = "<<mapped_label;
  }
 // sleep(20);



}

typedef std::pair<int, float> PAIR;
struct CmpByValue {
  bool operator()(const PAIR& lhs, const PAIR& rhs)
  {return lhs.second > rhs.second;}
 };



// template <typename Dtype>
// void ImageLabelDataLayer<Dtype>::sort_label_prob_map()
// {
  // //typedef std::pair<string, float> PAIR;


// }




template <typename Dtype>
void ImageLabelDataLayer<Dtype>::compute_label_skip_rate()
{
  //float rest_of_prob =0;
  float scale_factor =0;
  vector<PAIR> label_prob_vec(label_prob_map_.begin(), label_prob_map_.end());
  sort(label_prob_vec.begin(), label_prob_vec.end(), CmpByValue()); //prob descend order;


  float bottom_prob=label_prob_vec[num_top_label_balance_-1].second;
      //for(int i=0;i<num_top_label_balance_;i++)
       //LOG(INFO)<<"num_top_label_balance_["<< label_prob_vec[i].first<<"] = "<<label_prob_vec[i].second;
	  if(!ignore_rest_of_label_){
		  //for (int i=num_top_label_balance_;i<num_labels_with_prob_;++i)
		  //{
		//	rest_of_prob+=label_prob_vec[i].second;
		 // }
		//	rest_of_prob+=rest_of_label_prob_;
		    scale_factor =bottom_prob < rest_of_label_prob_? 1.0/bottom_prob: 1.0/rest_of_label_prob_;
	  }
	  else
	  {
		 scale_factor =1.0/bottom_prob ;

	  }
	  LOG(INFO)<<" scale_factor =  "<< scale_factor;
	  LOG(INFO)<<" bottom_prob =   " << bottom_prob;
	  LOG(INFO)<<"label_prob_vec.size = "<<label_prob_vec.size();
	 // sleep(10);

	  label_prob_map_.clear();
	  // remove the class that has prob lower that top k classes;
	  for(int i=0;i<num_top_label_balance_;++i)
	  {
	       int lb= label_prob_vec[i].first;
		   float prob =label_prob_vec[i].second;
		   label_prob_map_[lb]=prob;
		   // mapping the label based on freq
		   if(map2order_label_)
		   {
				label_mapping_map_[lb]=i;
		   }
	  }

	  // Init the rest of label class

	  for (int i=0;i<num_labels_ ;++i)
	  {
		int label =i;
		if(label_prob_map_.find(label)==label_prob_map_.end())
        {
			if(ignore_rest_of_label_){
				label_prob_map_[label]	=	0;
				}
			else
			   {
				 int rest_of_label =(num_labels_-num_top_label_balance_);
				 if(rest_of_label>0)
					label_prob_map_[label]	=	rest_of_label_prob_;///rest_of_label;
					 //LOG(INFO)<<"rest_of_label_prob["<<label<<"]=" <<label_prob_map_[label];
				}
			if(rest_of_label_mapping_){
				label_mapping_map_[label] =   rest_of_label_mapping_label_;
				//LOG(INFO)<<"rest_of_label_mapping_["<<label<<"]=" <<label_mapping_map_[label];
				}
			else
				label_mapping_map_[label] =   label;
			// if reorder label is set, override the rest_of_label_mapping_
			if(map2order_label_){
			    label_mapping_map_[label] =   num_top_label_balance_;
			}
		}
	 }
	 //sleep(20);
	  //auto iterSkipRate  =label_mapping_map_.begin();
	  std::map<int,float>::iterator iterProb;
      for (iterProb = label_prob_map_.begin(); iterProb !=label_prob_map_.end(); ++iterProb) {
				label_skip_rate_map_[iterProb->first] =ceil(iterProb->second*scale_factor);
				//LOG(INFO)<<"label_skip_rate_map_["<<iterProb->first<<"]=" <<label_skip_rate_map_[iterProb->first];


		}

}


template <typename Dtype>
bool ImageLabelDataLayer<Dtype>::accept_given_label(const int label)
{
		//
		//balancing_label_
		if(!balancing_label_)
		    return true;
		//LOG(INFO)<<"label_skip_rate_map_["<<label<<"] =" <<label_skip_rate_map_[label];
		if (label_skip_rate_map_[label] ==0)
		   return false;
		int reminder =PrefetchRand()%label_skip_rate_map_[label];
		if(reminder ==0)
		    return true;
		else
			return false;
}

template <typename Dtype>
int  ImageLabelDataLayer<Dtype>::get_converted_label(const int label){
         if(!balancing_label_)
		     return label;
	    else
			return label_mapping_map_[label];
}


template <typename Dtype>
unsigned int ImageLabelDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}



template <typename Dtype>
void ImageLabelDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
  // for(int i=0 ; i<threads_vec_.size();++i)
  // {
	// CHECK(!pthread_join(threads_vec_[i], NULL)) << "Pthread joining failed.";
  // }
}


template <typename Dtype>
void ImageLabelDataLayer<Dtype>::CreatePrefetchThread() {
  phase_ = Caffe::phase();
  //const bool prefetch_needs_rand = (phase_ == Caffe::TRAIN) &&
  //    (this->layer_param_.image_label_data_param().mirror() ||
   //    this->layer_param_.image_label_data_param().crop_size());

  // const bool prefetch_needs_rand = (this->layer_param_.image_label_data_param().mirror() ||
  //     this->layer_param_.image_label_data_param().crop_size());
  const bool prefetch_needs_rand = (this->layer_param_.image_label_data_param().mirror() ||
       is_crop_);
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }

   CHECK(!pthread_create(&thread_, NULL, ImageLabelDataLayerPrefetch<Dtype>,
        static_cast<void*>(this))) << "Pthread execution failed.";

  // Create the thread.
  //LOG(INFO)<<"create prefectch thread";

  //CreatePrefetchThread() muliple threads.
   // threads_vec_.clear();
   // layer_batchs_.clear();
   // int batch_size =this->layer_param_.image_label_data_param().batch_size();
   // int thread_num =1;
   // int par_batch_length =batch_size/thread_num;
   // int reminder   = batch_size%thread_num;
   // layer_batch_info_struct<Dtype> lbis;
  // for(int i=0;i<thread_num;++i){
	  // pthread_t thread_1;
	  // lbis.layer_pointer =this;
	  // lbis.batch_start   =i*par_batch_length;
	  // lbis.batch_end     =lbis.batch_start+par_batch_length;
	  // layer_batchs_.push_back(lbis);
	  // threads_vec_.push_back(thread_1);
	// }
  // if(reminder >0)
    // {
      // pthread_t thread_1;
	  // lbis.layer_pointer =this;
	  // lbis.batch_start   =thread_num*par_batch_length;
	  // lbis.batch_end     =batch_size;
	  // layer_batchs_.push_back(lbis);
	  // threads_vec_.push_back(thread_1);
    // }

	// for(int i=0;i<threads_vec_.size();++i)
	// {
		// CHECK(!pthread_create(&threads_vec_[i], NULL, ImageLabelDataLayerPrefetch<Dtype>,
			// static_cast<void*>(&layer_batchs_[i]))) << "Pthread execution failed.";
			// //sleep(0.01);
	// }


}


template <typename Dtype>
Dtype ImageLabelDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
   //LOG(INFO)<<"start forwarding data cd .....";
  JoinPrefetchThread();

  // Copy the data
  //LOG(INFO)<<"start copying data from prefech.....";
  caffe_copy(prefetch_data_->count(), prefetch_data_->cpu_data(),
             (*top)[0]->mutable_cpu_data());
 // LOG(INFO)<<"data copyied from prefech.....";
  if (output_labels_) {
    caffe_copy(prefetch_label_->count(), prefetch_label_->cpu_data(),
               (*top)[1]->mutable_cpu_data());
  }
  if(output_single_label_){
	 caffe_copy(prefetch_single_label_->count(), prefetch_single_label_->cpu_data(),
               (*top)[2]->mutable_cpu_data());
  }

  //LOG(INFO)<<"label copyied from prefech.....";
 // LOG(INFO)<<"Start a new prefetch thread";
  CreatePrefetchThread();
  return Dtype(0.);
}

INSTANTIATE_CLASS(ImageLabelDataLayer);
}  // namespace caffe
