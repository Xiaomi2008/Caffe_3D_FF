import sys, getopt

sys.path.insert(0, '/home/tzeng/caffe_3d/python')
sys.path.insert(0,'/home/tzeng/autoGenelable_multi_lables_proj/code/py-leveldb-read-only/build/lib.linux-x86_64-2.7')

import numpy as np
import hdf5storage
import leveldb
from leveldb import WriteBatch, LevelDB
import os
import imp
#import caffe
#sys.path.append('/home/tzeng/caffe_3d/python/caffe')
#foo = imp.load_source('caffe.io', '/home/tzeng/caffe_3d/python/caffe/__init__.py')
import caffe.io
from caffe.proto import caffe_pb2
print os.path.dirname(caffe_pb2.__file__) 

db_path_data='/home/tzeng/caffe_3d/data/test'
db_data=leveldb.LevelDB(db_path_data)	
datum = caffe_pb2.Datum()
mat_file='/home/tzeng/caffe_3d/data/test_out.mat'

window_num=0
for key in db_data.RangeIter(include_value = False):
  window_num=window_num+1

print window_num
n=0
# for key,value in db_data.RangeIter():
		# n=n+1
		# #f_size=len(value)
		# datum.ParseFromString(value)
		# f_size=len(datum.float_data)
		# if n>0:
		   # break

ft = np.zeros((1, 27))
lb = np.zeros((1, 27))
count=0
for key,value in db_data.RangeIter():
     datum.ParseFromString(value);
     #ft[count, :]=datum.data.tostring()
     lb[count,:]=datum.float_label
     count=count+1
print ft
data = {u'feat_label' : {
			u'feat' : ft,
			u'label' : lb,
		 }
	}

print 'save result to : %s' %(mat_file)
hdf5storage.savemat(mat_file,data, format='7.3') 
        