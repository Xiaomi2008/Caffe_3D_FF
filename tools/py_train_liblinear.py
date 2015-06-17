import sys, getopt

sys.path.insert(0, '/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/python')
sys.path.insert(0,'/home/tzeng/autoGenelable_multi_lables_proj/code/py-leveldb-read-only/build/lib.linux-x86_64-2.7')
import os
#import lmdb
import leveldb
#import scipy.io as sio
import numpy as np
import hdf5storage

import time
caffe_root = '/home/tzeng/autoGenelable_multi_lables_proj/code/caffe'
#sys.path.insert(os.path.join(caffe_root, 'python'))
import caffe.io
from caffe.proto import caffe_pb2
# db_path = '/home/rli/Downloaded_Software/caffe/examples/ISBI/foreground/train_lmdb_Consecutive_slices_right'
 
#db_path_label ='/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/vgg_16_fine_tune_gen/label_test'
#db_path_feats ='/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/vgg_16_fine_tune_gen/flat_conv5_1_eltmax_test'
#db_path = '/home/tzeng/ABA/autoGenelable_multi_lables_proj/data/all_test_slice_lvdb_ish'

def main(argv):
	db_path_label =''
	db_path_feats ='' 
	mat_file =''
	print argv
	try:
		opts, args = getopt.getopt(argv,"t:l:r:g:m",["test_label_db=","test_feature_db=","train_label_db=","train_feature_db=","mat_file="])
	except getopt.GetoptError:
		print 'feature_LDB_to_mat.py -t <test_label_db> -l <test_feature_db> -r <train_label_db> -g <train_feature_db> -m <output_mat_file>'
		sys.exit(2)
	
	print opts
	print args
	
		
	for opt, arg in opts:
		if opt in ("-t","--test_label_db"): 
			db_path_test_label=arg
		elif opt in("-l","--test_feature_db"):
			db_path_test_feats=arg
		if opt in ("-r","--train_label_db"): 
			db_path_train_label=arg
		elif opt in("-g","--train_feature_db"):
			db_path_train_feats=arg
		elif opt in("-m","--mat_file"):
			mat_file=arg
		#print arg+" "+opt

	#print(db_path_label)
	#print(db_path_feats)
	#print(mat_file)
    
   
	if not os.path.exists(db_path_test_label):
		raise Exception('db test label not found')
	if not os.path.exists(db_path_test_feats):
		raise Exception('db test feature not found')
	if not os.path.exists(db_path_train_label):
		print 'db_path_rain_label  is:'+ db_path_train_label
		raise Exception('db train label not found')
	if not os.path.exists(db_path_train_feats):
		raise Exception('db train feature not found')
		

	
	db_test_label=leveldb.LevelDB(db_path_test_label)
	db_test_feats=leveldb.LevelDB(db_path_test_feats)
	db_train_label=leveldb.LevelDB(db_path_train_label)
	db_train_feats=leveldb.LevelDB(db_path_train_feats)	
	#window_num =686
	datum = caffe_pb2.Datum()
	datum_lb = caffe_pb2.Datum()
	start=time.time();
	#ft = np.zeros((window_num, float(81)))
	#ft = np.zeros((window_num, float(100352)))
	#lb = np.zeros((window_num, float(81)))
	window_num=0
	for key in db_test_feats.RangeIter(include_value = False):
		window_num=window_num+1
	
	
	n=0
	for key,value in db_test_feats.RangeIter():
		n=n+1
		#f_size=len(value)
		datum.ParseFromString(value)
		f_size=len(datum.float_data)
		if n>0:
		   break
	n=0
	for key,value in db_test_label.RangeIter():
		n=n+1
		#l_size=len(value)
		datum.ParseFromString(value)
		l_size=len(datum.float_data)
		if n==1:
		   break
	te_ft = np.zeros((window_num, float(f_size)))
	te_lb = np.zeros((window_num, float(l_size)))
	

	window_num=0
	for key in db_train_feats.RangeIter(include_value = False):
		window_num=window_num+1
	n=0
	for key,value in db_train_feats.RangeIter():
		n=n+1
		#f_size=len(value)
		datum.ParseFromString(value)
		f_size=len(datum.float_data)
		if n>0:
		   break
	n=0
	for key,value in db_train_label.RangeIter():
		n=n+1
		#l_size=len(value)
		datum.ParseFromString(value)
		l_size=len(datum.float_data)
		if n==1:
		   break
	tr_ft = np.zeros((window_num, float(f_size)))
	tr_lb = np.zeros((window_num, float(l_size)))
	





	
	# for im_idx in range(window_num):
	count=0
	for key in db_test_feats.RangeIter(include_value = False):
	 datum.ParseFromString(db_test_feats.Get(key));
	 datum_lb.ParseFromString(db_test_label.Get(key));
	 te_ft[count, :]=datum.float_data
	 te_lb[count,:]=datum_lb.float_data
	 count=count+1
	 print 'convert feature # : %d key is %s' %(count,key)


	 count=0
	for key in db_train_feats.RangeIter(include_value = False):
	 datum.ParseFromString(db_train_feats.Get(key));
	 datum_lb.ParseFromString(db_train_label.Get(key));
	 tr_ft[count, :]=datum.float_data
	 tr_lb[count,:]=datum_lb.float_data
	 count=count+1

	print 'time 1: %f' %(time.time() - start)
	prob  = problem(tr_lb[1,:], tr_ft)
	m = train(prob, '-c 4')
	p_label, p_acc, p_val = predict(te_lb[:], te, m)
	print 'done!'

if __name__ == "__main__":
   main(sys.argv[1:])