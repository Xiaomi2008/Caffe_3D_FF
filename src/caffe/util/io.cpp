// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <string>
#include <vector>
#include <fstream>  // NOLINT(readability/streams)
#include <utility>

#include "caffe/common.hpp"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"

using std::fstream;
using std::ios;
using std::max;
using std::string;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

namespace caffe {

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  //coded_input->SetTotalBytesLimit(1073741824, 536870912);
  coded_input->SetTotalBytesLimit(1e9, 9e8);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

bool ReadImage(const string& filename,
    const int height, const int width, const bool is_color, Datum* datum) {
  cv::Mat cv_img;

  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  if (height > 0 && width > 0) {
    cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
    cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
  } else {
    cv_img = cv::imread(filename, cv_read_flag);
  }
  if (!cv_img.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return false;
  }
  int num_channels = (is_color ? 3 : 1);
  datum->set_channels(num_channels);
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  string* datum_string = datum->mutable_data();
  if (is_color) {
    for (int c = 0; c < num_channels; ++c) {
      for (int h = 0; h < cv_img.rows; ++h) {
        for (int w = 0; w < cv_img.cols; ++w) {
          datum_string->push_back(
            static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
        }
      }
    }
  } else {  // Faster than repeatedly testing is_color for each pixel w/i loop
    for (int h = 0; h < cv_img.rows; ++h) {
      for (int w = 0; w < cv_img.cols; ++w) {
        datum_string->push_back(
          static_cast<char>(cv_img.at<uchar>(h, w)));
        }
      }
  }
  return true;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color, Datum* datum) {

  if (ReadImage(filename, height, width, is_color, datum)) {
    if (datum->label_size() > 0) {
      datum->set_label(0, label);
    } else {
      datum->add_label(label);
    }
    return true;
  } else {
    return false;
  }
}

bool ReadImageToDatum(const string& filename, const std::vector<int> labels,
    const int height, const int width, const bool is_color, Datum* datum) {

  if (labels.size() > 0) {
    LOG(INFO)<<"using io.cpp ReadImageToDatum  reading image file is : "<< filename ;
    if (ReadImageToDatum(filename, labels[0],
                         height, width, is_color, datum)) {
      for (int i = 1 ; i < labels.size(); ++i) {
        if (datum->label_size() <= i) {
          datum->add_label(labels[i]);
        } else {
          datum->set_label(i, labels[i]);
        }
      }
      return true;
    } else {
      return false;
    }
  } else {
    return ReadImage(filename, height, width, is_color, datum);
  }
}


bool ReadImage2CvObject(const string& filename,const int height, const int width,const bool is_color, cv::Mat& cv_img)
{ 

 int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
 if (height > 0 && width > 0) {
    cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
    cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
  } else {
    cv_img = cv::imread(filename, cv_read_flag);
  }
  if (!cv_img.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return false;
  }
}






// // write multiple images such as  brain slices  to one datum acting like multiple channels
bool ReadImage(const std::vector<string>& filenames,
    const int height, const int width, const bool is_color, Datum* datum) {
	
  
  cv::Mat cv_img;
   
  // int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    // CV_LOAD_IMAGE_GRAYSCALE);
  // if (height > 0 && width > 0) {
    // cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
    // cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
  // } else {
    // cv_img = cv::imread(filename, cv_read_flag);
  // }
  // if (!cv_img.data) {
    // LOG(ERROR) << "Could not open or find file " << filename;
    // return false;
  // }
  
  
  if(!ReadImage2CvObject(filenames[0],height, width,is_color, cv_img)){
	 LOG(ERROR) << "Could not open or find file " << filenames[0];
     return false;
  }
  int num_files    =filenames.size();
  int num_channels = (is_color ? 3 : 1);
  datum->set_channels(num_channels*num_files);
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  
 // datum->set_depth(num_files);
  
  datum->clear_data();
  datum->clear_float_data();
  string* datum_string = datum->mutable_data();
  if (is_color) {
	for (int f=0; f<num_files; ++f){
		   cv::Mat cv_img2;
		  if(!ReadImage2CvObject(filenames[f],height, width,is_color, cv_img2)){
			 LOG(ERROR) << "Could not open or find file " << filenames[0];
			 return false;
		  }
		for (int c = 0; c < num_channels; ++c) {
		  for (int h = 0; h < cv_img2.rows; ++h) {
			for (int w = 0; w < cv_img2.cols; ++w) {
			  datum_string->push_back(
				static_cast<char>(cv_img2.at<cv::Vec3b>(h, w)[c]));
			}
		  }
		}
	}
  } else {  // Faster than repeatedly testing is_color for each pixel w/i loop
	for (int f=0; f<num_files; ++f){
		 cv::Mat cv_img2;
		  if(!ReadImage2CvObject(filenames[f],height, width,is_color, cv_img2)){
			 LOG(ERROR) << "Could not open or find file " << filenames[0];
			 return false;
		  }
		for (int h = 0; h < cv_img2.rows; ++h) {
			for (int w = 0; w < cv_img2.cols; ++w) {
			datum_string->push_back(
			static_cast<char>(cv_img2.at<uchar>(h, w)));
			}
		}
	}
  }
  return true;
}



// // write multiple images such as  brain slices  to one datum acting 3d data
bool Read3DImage(const std::vector<string>& filenames,
    const int height, const int width, const bool is_color, Datum* datum) {
	
  
  cv::Mat cv_img;
   
  
  
  if(!ReadImage2CvObject(filenames[0],height, width,is_color, cv_img)){
	 LOG(ERROR) << "Could not open or find file " << filenames[0];
     return false;
  }
  int num_files    =filenames.size();
  int num_channels = (is_color ? 3 : 1);
  datum->set_channels(num_channels);
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->set_depth(num_files);
  
  datum->clear_data();
  datum->clear_float_data();
  string* datum_string = datum->mutable_data();
  
  std::vector<cv::Mat> cv_imgs_vec;
  if (is_color) {
	for (int f=0; f<num_files; ++f){
		   cv::Mat cv_img2;
		  if(!ReadImage2CvObject(filenames[f],height, width,is_color, cv_img2)){
			 LOG(ERROR) << "Could not open or find file " << filenames[0];
			 return false;
		  }
		 cv_imgs_vec.push_back(cv_img2);
		}
	   //LOG(INFO) <<"vector size = " << cv_imgs_vec.size(); 
		
		for (int c = 0; c < num_channels; ++c) {
			for (int h = 0; h < cv_img.rows; ++h) {
				for (int w = 0; w < cv_img.cols; ++w) {
					for (int f=0; f<num_files; ++f){
						datum_string->push_back(
						static_cast<char>(cv_imgs_vec[f].at<cv::Vec3b>(h, w)[c]));
					}
				}
			}
		}
		
		
		
  } else {  // Faster than repeatedly testing is_color for each pixel w/i loop
		for (int f=0; f<num_files; ++f){
			 cv::Mat cv_img2;
			  if(!ReadImage2CvObject(filenames[f],height, width,is_color, cv_img2)){
				 LOG(ERROR) << "Could not open or find file " << filenames[0];
				 return false;
			  }
			  cv_imgs_vec.push_back(cv_img2);
			}
			
		for (int h = 0; h < cv_img.rows; ++h) {
			for (int w = 0; w < cv_img.cols; ++w) {
				for (int f=0; f<num_files; ++f){
					datum_string->push_back(
					static_cast<char>(cv_imgs_vec[f].at<uchar>(h, w)));
				}
			}
		}
	}
  return true;
}






// // This funcion is made for multiple images to store in one datum as if they are 3d image in the order of depth-width-height-channel) , by Tao Zeng at ODU march 29 2015
bool ReadImageToDatum(const std::vector<string> filenames, const std::vector<int> labels,
    const int height, const int width, const bool is_color, Datum* datum) {

  //if (labels.size() > 0) {
    //LOG(INFO)<<"using io.cpp ReadImageToDatum  reading image file is : "<< filename ;
   // if (ReadImageToDatum(filename, labels[0],
     //                    height, width, is_color, datum)) 
						 
	if ( Read3DImage(filenames, height, width, is_color,datum)	){
      for (int i = 0 ; i < labels.size(); ++i) {
        if (datum->label_size() <= i) {
          datum->add_label(labels[i]);
        } else {
          datum->set_label(i, labels[i]);
        }
      }
      return true;
    } else {
      return false;
    }
  //} else {
  //  return ReadImage(filename, height, width, is_color, datum);
  //}
}

// void ReadImagesList(const string& source,
    // std::vector<std::pair<std::vector<string>, std::vector<std::vector<int> > > >* images_vec, std::vector<string>& sectionIDs){
		// images_vec->clear();
	  // sectionIDs.clear();
	  // ISH_section_param ISHsectionIMinfo;
	  // ReadProtoFromTextFileOrDie(source, &ISHsectionIMinfo);
	 
	  
	  // int isize =ISHsectionIMinfo.ish_section_info_size();
	   // LOG(INFO)<<"there are "<< isize<<" of sections:";
	  // for (int section_idx = 0; section_idx < ISHsectionIMinfo.ish_section_info_size(); ++section_idx) 
	  // {
		 // std::vector<string> filenames;
		 // std::vector<std::vector<int> labels>;
		 // filenames.clear();
		 // labels.clear();
		// const ISH_section& section_param = ISHsectionIMinfo.ish_section_info(section_idx);
		 // int secid =section_param.sectionid();
		
		 // std::stringstream s;
		 // s << secid;
		 // string secidst(s.str());
		 // sectionIDs.push_back(secidst);
		  
		 // LOG(INFO)<<"section ID = "<<secid;
		// for(int im_idx=0;im_idx<section_param.image_size();++im_idx)
		// {
			// const imageInfo& im_param = section_param.image(im_idx);
				// LOG(INFO)<<"image path = "<<im_param.filename();
				// filenames.push_back(im_param.filename());
		// }
		// //string labels=section_param.labels();
		
		// std::istringstream iss(section_param.labels());
		// int label;
		// while (iss >> label) {
		  // labels.push_back(label);
		  // //LOG(INFO)<<label;
		// }
		
		// images_vec->push_back(std::make_pair(filenames, labels));
	  // }
// }



void ReadImagesList(const string& source,
    std::vector<std::pair<std::vector<string>, std::vector<int> > >* images_vec,std::vector<string>& sectionIDs){

  images_vec->clear();
  sectionIDs.clear();
  ISH_section_param ISHsectionIMinfo;
  ReadProtoFromTextFileOrDie(source, &ISHsectionIMinfo);
 
  
  int isize =ISHsectionIMinfo.ish_section_info_size();
   LOG(INFO)<<"there are "<< isize<<" of sections:";
  for (int section_idx = 0; section_idx < ISHsectionIMinfo.ish_section_info_size(); ++section_idx) 
  {
     std::vector<string> filenames;
     std::vector<int> labels;
	 filenames.clear();
	 labels.clear();
	const ISH_section& section_param = ISHsectionIMinfo.ish_section_info(section_idx);
	 int secid =section_param.sectionid();
	
	 std::stringstream s;
     s << secid;
	 string secidst(s.str());
	 sectionIDs.push_back(secidst);
	  
	 LOG(INFO)<<"section ID = "<<secid;
	for(int im_idx=0;im_idx<section_param.image_size();++im_idx)
	{
		const imageInfo& im_param = section_param.image(im_idx);
		    LOG(INFO)<<"image path = "<<im_param.filename();
			filenames.push_back(im_param.filename());
	}
	//string labels=section_param.labels();
	
	std::istringstream iss(section_param.labels());
	int label;
	while (iss >> label) {
      labels.push_back(label);
	  //LOG(INFO)<<label;
    }
	
	images_vec->push_back(std::make_pair(filenames, labels));
  }
}


void ReadImagesList(const string& source,
    std::vector<std::pair<std::string, std::vector<int> > >* images_vec) {
  // Read the file with filenames and labels
  LOG(INFO) << "Opening file using  ReadImagesList from: " << source;
  std::ifstream infile(source.c_str());
  CHECK(infile) << "Error opening file";
  std::string line;
  int line_num = 1;
  int num_labels = 0;
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    string filename;
    std::vector<int> labels;
    int label;
    CHECK(iss >> filename) << "Error reading line " << line_num;
    while (iss >> label) {
      labels.push_back(label);
    }
    if (line_num == 1) {
      // Use first line to set the number of labels
      num_labels = labels.size();
    }
    CHECK_EQ(labels.size(), num_labels) <<
      filename << " error at line " << line_num << std::endl <<
      " All images should have the same number of labels";
    line_num++;
    images_vec->push_back(std::make_pair(filename, labels));
  }
  LOG(INFO) << "Read " << line_num - 1 << " images with " <<
    num_labels << " labels";
}




// Verifies format of data stored in HDF5 file and reshapes blob accordingly.
template <typename Dtype>
void hdf5_load_nd_dataset_helper(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob) {
  // Verify that the number of dimensions is in the accepted range.
  herr_t status;
  int ndims;
  status = H5LTget_dataset_ndims(file_id, dataset_name_, &ndims);
  CHECK_GE(ndims, min_dim);
  CHECK_LE(ndims, max_dim);

  // Verify that the data format is what we expect: float or double.
  std::vector<hsize_t> dims(ndims);
  H5T_class_t class_;
  status = H5LTget_dataset_info(
      file_id, dataset_name_, dims.data(), &class_, NULL);
  CHECK_GE(status, 0) << "Failed to get dataset info for " << dataset_name_;
  CHECK_EQ(class_, H5T_FLOAT) << "Expected float or double data";

  blob->Reshape(
    dims[0],
    (dims.size() > 1) ? dims[1] : 1,
    (dims.size() > 2) ? dims[2] : 1,
    (dims.size() > 3) ? dims[3] : 1);
}

template <>
void hdf5_load_nd_dataset<float>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<float>* blob) {
  hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
  herr_t status = H5LTread_dataset_float(
    file_id, dataset_name_, blob->mutable_cpu_data());
  CHECK_GE(status, 0) << "Failed to read float dataset " << dataset_name_;
}

template <>
void hdf5_load_nd_dataset<double>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<double>* blob) {
  hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
  herr_t status = H5LTread_dataset_double(
    file_id, dataset_name_, blob->mutable_cpu_data());
  CHECK_GE(status, 0) << "Failed to read double dataset " << dataset_name_;
}

template <>
void hdf5_save_nd_dataset<float>(
    const hid_t file_id, const string dataset_name, const Blob<float>& blob) {
  hsize_t dims[HDF5_NUM_DIMS];
  dims[0] = blob.num();
  dims[1] = blob.channels();
  dims[2] = blob.height();
  dims[3] = blob.width();
  herr_t status = H5LTmake_dataset_float(
      file_id, dataset_name.c_str(), HDF5_NUM_DIMS, dims, blob.cpu_data());
  CHECK_GE(status, 0) << "Failed to make float dataset " << dataset_name;
}

template <>
void hdf5_save_nd_dataset<double>(
    const hid_t file_id, const string dataset_name, const Blob<double>& blob) {
  hsize_t dims[HDF5_NUM_DIMS];
  dims[0] = blob.num();
  dims[1] = blob.channels();
  dims[2] = blob.height();
  dims[3] = blob.width();
  herr_t status = H5LTmake_dataset_double(
      file_id, dataset_name.c_str(), HDF5_NUM_DIMS, dims, blob.cpu_data());
  CHECK_GE(status, 0) << "Failed to make double dataset " << dataset_name;
}

}  // namespace caffe
