import sys, getopt

sys.path.insert(0, '/home/tzeng/caffe_flx_kernel/python')
#sys.path.insert(0,'/home/tzeng/autoGenelable_multi_lables_proj/code/py-leveldb-read-only/build/lib.linux-x86_64-2.7')
import os
#import lmdb
import leveldb
#import scipy.io as sio
import numpy as np
import hdf5storage

import time
#caffe_root = '/home/tzeng/caffe_3d/python'
#sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
print os.path.dirname(caffe.__file__) 
from caffe.proto import caffe_pb2
# db_path = '/home/rli/Downloaded_Software/caffe/examples/ISBI/foreground/train_lmdb_Consecutive_slices_right'
 
db_path_label ='/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/vgg_16_fine_tune_gen/label_test'
db_path_feats ='/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/vgg_16_fine_tune_gen/flat_conv5_1_eltmax_test'
#db_path = '/home/tzeng/ABA/autoGenelable_multi_lables_proj/data/all_test_slice_lvdb_ish'

def main(argv):
	#db_path_label =''
	db_path_feats ='' 
	mat_file =''
	print argv
	#try:
	opts, args = getopt.getopt(argv,'f:o:h',['feature_db=','mat_file=','help'])
	#except getopt.GetoptError:
	#	print 'feature_LDB_to_mat.py  -f <feature_db> -m <output_mat_file>'
	#	sys.exit(2)
	
	print opts
	print args
	
	
	for opt, arg in opts:
		if opt in("-f","--feature_db"):
			db_path_feats=arg
		elif opt in("-o","--mat_file"):
			mat_file=arg
		elif opt in('-h', '--help'):
			sys.exit(2)
		print arg+" "+opt

	#print(db_path_label)
	print(db_path_feats)
	print(mat_file)

	#if not os.path.exists(db_path_label):
	#	raise Exception('db label not found')
	if not os.path.exists(db_path_feats):
		raise Exception('db feature not found')
		

	
	#db_label=leveldb.LevelDB(db_path_label)
	db_feats=leveldb.LevelDB(db_path_feats)	
	#window_num =686
	datum = caffe_pb2.Datum()
	#datum_lb = caffe_pb2.Datum()
	start=time.time();
	#ft = np.zeros((window_num, float(81)))
	#ft = np.zeros((window_num, float(100352)))
	#lb = np.zeros((window_num, float(81)))
	window_num=0
	for key in db_feats.RangeIter(include_value = False):
		window_num=window_num+1
	
	f_size =1
	n=0
	for key,value in db_feats.RangeIter():
		#f_size=len(value)
		datum.ParseFromString(value)
		f_size=len(datum.float_data)
		if f_size==0:
	 		f_size=len(datum.data)
        #f_size=len(datum.data)
        #break
	print('data size is %d', f_size)
	#for key,value in db_label.RangeIter():
	#	n=n+1
		#l_size=len(value)
	#	datum.ParseFromString(value)
	#	l_size=len(datum.float_data)
	#	if n==1:
	#	   break
	ft = np.zeros((window_num, int(f_size)))
	#lb = np.zeros((window_num, float(l_size)))
	
	
	# for im_idx in range(window_num):
	count=0
	for key in db_feats.RangeIter(include_value = False):
	 #datum.ParseFromString(db_label.Get('%d' %(im_idx)))
	 datum.ParseFromString(db_feats.Get(key));
	 print('%d %d %d %d', datum.channels, datum.height, datum.width, datum.depth) 
	 #datum_lb.ParseFromString(db_label.Get(key));
	 #datum.ParseFromString(db_feats.Get('%d' %(im_idx)))
	 #ft[im_idx, :]=caffe.io.datum_to_array(datum)
	 ft=datum.float_data
	 print len(ft)
	 if len(ft)==0:
	 	ft=datum.data
	 d_dim =len(ft)/(datum.width * datum.depth)
	 print d_dim
	 #ft[count, :]=datum.data
	 #ft=np.fromstring(datum.data, dtype = np.int).reshape(datum.width, datum.depth)
     #datum.channels, datum.height, datum.width, datum.depth
	 #lb[count,:]=datum_lb.float_data
	 count=count+1
	 #ft=np.reshape(ft,len(ft), order ='F')
	 ft=np.reshape(ft,(d_dim,datum.width, datum.depth))
	 #ft=np.reshape(ft,(12,400, 400))
     # b.reshape(2,3,order='F')
	 #print key
	 print 'convert feature # : %d key is %s' %(count,key)
	print 'time 1: %f' %(time.time() - start)

	data = {u'feat_label' : {
			u'feat' : ft,
			#u'label' : lb,
		 }
	 }
	print 'save result to : %s' %(mat_file)
	hdf5storage.savemat(mat_file,data, format='7.3') 
	print 'time 2: %f' %(time.time() - start)
	print 'done!'

if __name__ == "__main__":
   main(sys.argv[1:])